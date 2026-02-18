from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.ollama import OllamaProvider

# Define our structured response format
class CityInfo(BaseModel):
    city: str = Field(description="The name of the city")
    country: str = Field(description="The country of the city")
    population: int = Field(description="The population of the city")
    fun_fact: str = Field(description="A fun fact about the city")
    current_weather: str = Field(description="The weather right now in the city")
    miscellaneous: str = Field(description="Any other information about the city, that is not covered by the other fields")
    
# 1. Define the Ollama model using the OpenAI-compatible interface
# Note: Ollama's local URL is typically http://localhost:11434/v1
model = OpenAIChatModel(
    model_name='qwen3',
    provider=OllamaProvider(base_url='http://localhost:11434/v1'),
)

# 2. Initialize the Agent
agent = Agent(
    model=model,
    output_type=CityInfo,
    system_prompt="Extract city information. Respond in the format of the CityInfo model. For the weather info - use the provided get_weather tool."
)

# Define a tool the agent can call
@agent.tool_plain
def get_weather(city: str) -> str:
    """Gets the current weather for a city."""
    return f"The weather in {city} is currently 22°C and sunny."

# 3. Run the agent
result = agent.run_sync("Tell me about Tokyo. Also check the current weather there.")
print(f"Result: {result}")
print(f"City: {result.output.city}")
print(f"Fact: {result.output.fun_fact}")
print(f"Current Weather: {result.output.current_weather}")
