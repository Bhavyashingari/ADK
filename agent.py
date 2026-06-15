from google.adk.agents import Agent
from google.adk.code_executors import BuiltInCodeExecutor
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.adk.tools import google_search_tool
from google.adk.tools.mcp_tool import McpToolset
from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams
from google.adk.tools.mcp_tool.mcp_session_manager import SseConnectionParams
from mcp import StdioServerParameters
from google.genai import types
from pydantic import BaseModel, Field


class Output(BaseModel):
    response: str = Field(..., description="The response from the agent.")

MODEL = "gemini-3.1-flash"
mcp_toolset = McpToolset(
    connection_params=StdioConnectionParams(
        server_params=StdioServerParameters(
            command=["npx"],
            args=["-y", "@placeholder/mcp-server"],
            tool_filter=["list_directory", "read_file"]
        )
    )
)

connection_params = SseConnectionParams(
    url="https://mcp-server.example.com",
    headers={"Authorization": "Bearer <token>"},
)

safety_config = types.SafetyConfig(
    content_filter=types.ContentFilterConfig(
        filter_categories=[
            types.ContentFilterCategory.HATE_SPEECH,
            types.ContentFilterCategory.VIOLENCE,
            types.ContentFilterCategory.SEXUALLY_EXPLICIT,
        ],  
        threshold=types.ContentFilterThreshold.HIGH,
    )
)


greet_agent = Agent(
    name="greet_agent",
    description="An agent that greets the user.",
    generate_content_config=types.GenerateContentConfig(
        temperature=0.7,
        max_output_tokens=100,
        safety_settings=safety_config,
    ),
    output_schema=Output,
    model=MODEL,
    tools=[google_search_tool],
    code_executor=BuiltInCodeExecutor(),
    output_key="topic"
)

session_service = InMemorySessionService()
session = session_service.create_session(app_name="session1", user_id="user1")
runner = Runner(app_name="session1", session_service=session_service, agent=greet_agent)

validation = session.state["temp:current_step"] = "Validation"
topic = session.state["conversation_topic"] = "item out of stock"
session.state["user:language"] = "English"
session.state["app:version"] = "18.80.0"

result = runner.run(user_id="user1", session_id=session.id, new_message=types.Content(parts=[types.Part(text="Welcome to Zomato")]))
user_name = session.state.get("user_name", "Guest")
chat_count = session.state.get("chat_count", 0)

for event in result:
    if event.is_final_response():
        print(f"Agent response: {event.final_response().content.parts[0].text}")