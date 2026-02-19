import os
from pydantic_ai import Agent
from openai import AsyncAzureOpenAI
from pydantic_ai.mcp import load_mcp_servers
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

api_token = os.environ.get('PYDANTIC_AI_GATEWAY_API_KEY')

# 1. Initialize the client, server and the Agent
azure_client = AsyncAzureOpenAI(
    azure_endpoint="https://genai-training-track-q12026-team-6.openai.azure.com/",
    api_key=api_token,
    api_version="2024-10-21",      # Is this the latest version?
)
servers = load_mcp_servers('mcp_config.json')
model = OpenAIChatModel(
    model_name='gpt-4.1',
    provider=OpenAIProvider(openai_client=azure_client)
)
agent = Agent(
    model=model,
    toolsets=servers,
    system_prompt="You are a helpful assistant that can answer questions and help with tasks. When possible, use the provided tools in the agent and the MCP servers to get the best answer."
)

# 2. Define a tool that the agent can call, in addition to the tools provided by the MCP server
@agent.tool_plain
def get_weather(city: str) -> str:
    """Gets the current weather for a city."""
    return f"The weather in {city} is currently 22°C and sunny."

# 3. Run the agent
result = agent.run_sync("Tell me about Tokyo.")
print(f"Result: {result.output}")
# Pass message_history so the agent knows "there" refers to Tokyo
result = agent.run_sync(
    "What's the weather there like today?",
    message_history=result.new_messages(),
)
print(f"--------------------------------")
print(f"Result: {result.output}")
result = agent.run_sync("How much is 372 plus 454, please?")
print(f"--------------------------------")
print(f"Result: {result.output}")
