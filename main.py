from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.ollama import OllamaProvider

# 1. Define the Ollama model using the OpenAI-compatible interface
# Note: Ollama's local URL is typically http://localhost:11434/v1
model = OpenAIChatModel(
    model_name='llama3.1',
    provider=OllamaProvider(base_url='http://localhost:11434/v1'),
)

# 2. Initialize the Agent
agent = Agent(model=model, system_prompt="You are a helpful, concise assistant.")

# 3. Run the agent
result = agent.run_sync("What are the benefits of local LLMs?")
print(result.output)
