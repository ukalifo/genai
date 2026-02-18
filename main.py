from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.mcp import MCPServerStreamableHTTP
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.ollama import OllamaProvider
    
# 1. Define the Ollama model using the OpenAI-compatible interface
# Note: Ollama's local URL is typically http://localhost:11434/v1
model = OpenAIChatModel(
    model_name='qwen3',
    provider=OllamaProvider(base_url='http://localhost:11434/v1'),
)

# 2. Initialize the server and the Agent
server = MCPServerStreamableHTTP('http://localhost:8000/mcp')
agent = Agent(
    model=model,
    # toolsets=[server.toolset()],
    toolsets=[server],
    system_prompt="You are a helpful assistant that can answer questions and help with tasks. When possiböe, use the provided tools in the agent and the MCP servers to get the best answer."
)

# Define a tool the agent can call
@agent.tool_plain
def get_weather(city: str) -> str:
    """Gets the current weather for a city."""
    return f"The weather in {city} is currently 22°C and sunny."

# 3. Run the agent
result = agent.run_sync("Tell me about Tokyo. Also check the current weather there.")
print(f"Result: {result.output}")
result = agent.run_sync("How much is 72 plus 54.")
print(f"--------------------------------")
print(f"Result: {result.output}")
