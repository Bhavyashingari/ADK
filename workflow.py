from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.core.tools import FunctionTool
from llama_index.core.agent.workflow import AgentWorkflow, AgentInput, AgentOutput, ToolCallResult, FunctionAgent
from llama_index.core.workflow import Context, Event, StartEvent, StopEvent, Workflow, step
from tavily import AsyncTavilyClient
import os

async def search_web(query: str) -> str:
    """answer the query using Tavily's web search API."""
    client = AsyncTavilyClient(api_key=os.environ.get("TAVILY_API_KEY"))
    return str(await client.search(query=query))

async def set_user(ctx: Context, user_name: str) -> str:
    """Set the user name in the context."""
    state = ctx.get("state")
    state["user_name"] = user_name
    await ctx.set("state", state)
    return f"User name set to {user_name}"

search_tool = FunctionTool.from_defaults(fn=search_web)
llm = AzureOpenAI(
    model='gpt-4o',
    api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.environ.get("AZURE_OPENAI_API_BASE"),
    api_version=os.environ.get("AZURE_OPENAI_API_VERSION"),
)

stateful_workflow = AgentWorkflow.from_tools_or_functions(
    [search_tool, set_user],
    llm=llm,
    system_prompt="You are a helpful assistant that can answer questions using web search.",
)

stateful_workflow_context = Context(stateful_workflow)
response = stateful_workflow.run("What is the capital of France?", ctx={"user_name": "Alice"})

for event in response.stream_events():
    if isinstance(event, AgentInput):
       print("Agent input: ", event.input)  # Current input messages
       print("Agent name:", event.current_agent_name)  # Current agent name
    elif isinstance(event, AgentOutput):
       print("Agent output: ", event.response)  # Current full response
       print("Tool calls made: ", event.tool_calls)  # Selected tool calls, if any
       print("Raw LLM response: ", event.raw)  # Raw llm api response
    elif isinstance(event, ToolCallResult):
       print("Tool called: ", event.tool_name)  # Tool name
       print("Arguments to the tool: ", event.tool_kwargs)  # Tool kwargs
       print("Tool output: ", event.tool_output)  # Tool output



class FeedbackEvent(Event):
    feedback: str

class ReviewEvent(Event):
    report: str

class GenerateEvent(Event):
    research_topic: str

class QuestionEvent(Event):
    question: str

class AnswerEvent(Event):
    question: str
    answer: str

class ProgressEvent(Event):
    msg: str

question_agent = FunctionAgent(
    tools=[],
    llm=llm,
    verbose=False,
    system_prompt="""You are part of a deep research system.
      Given a research topic, you should come up with a bunch of questions
      that a separate agent will answer in order to write a comprehensive
      report on that topic. To make it easy to answer the questions separately,
      you should provide the questions one per line. Don't include markdown
      or any preamble in your response, just a list of questions."""
)

answer_agent = FunctionAgent(
    tools=[search_web],
    llm=llm,
    verbose=False,
    system_prompt="""You are part of a deep research system.
      Given a specific question, your job is to come up with a deep answer
      to that question, which will be combined with other answers on the topic
      into a comprehensive report. You can search the web to get information
      on the topic, as many times as you need."""
)
report_agent = FunctionAgent(
    tools=[],
    llm=llm,
    verbose=False,
    system_prompt="""You are part of a deep research system.
      Given a set of answers to a set of questions, your job is to combine
      them all into a comprehensive report on the topic."""
)

review_agent = FunctionAgent(
    tools=[],
    llm=llm,
    verbose=False,
    system_prompt="""You are part of a deep research system.
      Your job is to review a report that's been written and suggest
      questions that could have been asked to produce a more comprehensive
      report than the current version, or to decide that the current
      report is comprehensive enough."""
)


class DeepResearchWorkflow(Workflow):
    @step
    async def setup(self, ctx: Context, ev: StartEvent) -> GenerateEvent:
        self.question_agent = ev.question_agent
        self.answer_agent = ev.answer_agent
        self.report_agent = ev.report_agent
        self.review_agent = ev.review_agent
        self.review_cycles = 0

        ctx.write_event_to_stream(ProgressEvent(msg="Starting deep research workflow"))
        return GenerateEvent(research_topic=ev.research_topic)
    
    @step
    async def generate_questions(self, ctx: Context, ev: GenerateEvent | FeedbackEvent) -> QuestionEvent:
        await ctx.set("research_topic", ev.research_topic)
        ctx.write_event_to_stream(ProgressEvent(msg=f"Research topic: {ev.research_topic}"))
        prompt = f"Generate a list of questions to research the topic: {ev.research_topic}"
        if isinstance(ev, FeedbackEvent):
            ctx.write_event_to_stream(ProgressEvent(msg=f"Feedback received: {ev.feedback}"))
            prompt += f"You have previously researched on this topic and received the following feedback: {ev.feedback}. Please generate a new list of questions that addresses this feedback."

        result = await self.question_agent.run(prompt)
        lines = result.response.split()
        questions = [line.strip() for line in lines if line.strip()]
        await ctx.set("total_questions", len(questions))
        for question in questions:
            ctx.send_event(QuestionEvent(question=question))

    @step
    async def answer_questions(self, ctx: Context, ev: QuestionEvent) -> AnswerEvent:
        result = await self.answer_agent.run(user_msg=f"Research the answer to the following question: {ev.question}"
                                             "You can use web search to find information to answer the question.")
        ctx.write_event_to_stream(ProgressEvent(msg=f"Answered question: {ev.question}"))
        return AnswerEvent(question=ev.question, answer=result.response)
    
    @step
    async def write_report(self, ctx: Context, ev: AnswerEvent) -> ReviewEvent:
        answers = await ctx.collect_events(ev, [AnswerEvent] * await ctx.get("total_questions"))
        if answers is None:
            ctx.write_event_to_stream(ProgressEvent(msg="Waiting for all answers to be collected..."))
            return None
        ctx.write_event_to_stream(ProgressEvent(msg="All answers collected, writing report..."))
        all_answers = ""
        for q_and_a in answers:
            all_answers += f"Question: {q_and_a.question}\nAnswer: {q_and_a.answer}\n\n"
        
        result = await self.report_agent.run(user_msg=f"Write a comprehensive report on the topic: {await ctx.get('research_topic')}\n\n{all_answers}")
        ctx.write_event_to_stream(ProgressEvent(msg="Report written, reviewing report..."))
        return ReviewEvent(report=result.response)
    
    @step
    async def review_report(self, ctx: Context, ev: ReviewEvent) -> StopEvent | FeedbackEvent:
        answer = self.review_agent.run(user_msg=f"Review the following report and suggest questions that could have been asked to produce a more comprehensive report, or decide that the current report is comprehensive enough if yes return the only string ACCEPTABLE else return REVIEW_REQUIRED:\n\n{ev.report}")
        self.review_cycles +=1
        if str(answer) == "ACCEPTABLE" or self.review_cycles >= 3:
            ctx.write_event_to_stream(ProgressEvent(msg="Report is acceptable, stopping workflow."))
            return StopEvent()
        else:
            ctx.write_event_to_stream(ProgressEvent(msg="Report requires review, generating new questions."))
            return FeedbackEvent(feedback=str(answer))
        

