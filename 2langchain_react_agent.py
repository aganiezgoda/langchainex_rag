# ReAct Agent with LangChain
# https://python.langchain.com/docs/concepts/agents/#react-agents
# https://python.langchain.com/docs/how_to/migrate_agent/

"""
ReAct (Reasoning + Acting) pattern:
The agent follows a loop of:  Thought -> Action -> Observation -> Thought -> ...

1. THOUGHT:  The LLM reasons about the current state and decides what to do next.
2. ACTION:   The LLM picks a tool and provides arguments for it.
3. OBSERVATION: The tool is executed and returns a result.
4. Repeat until the LLM decides it has enough info to produce a final answer.

This is different from a simple chain because:
- The agent decides WHICH tools to use and HOW MANY TIMES.
- The agent can reason about intermediate results before deciding next steps.
- There is no fixed sequence — the flow is dynamic based on the problem.

Below we build a ReAct agent with 3 simple tools:
  - get_weather: returns fake weather for a city
  - multiply:    multiplies two numbers
  - get_word_length: returns the length of a word

We then ask questions that require the agent to reason and pick the right tool(s).
"""

from langchain_openai import AzureChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent

from azure.identity import DefaultAzureCredential, get_bearer_token_provider

from dotenv import load_dotenv
import os

load_dotenv()

# ── Azure OpenAI setup ──────────────────────────────────────────────

OPENAI_API_VERSION = os.getenv("OPENAI_API_VERSION")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_DEPLOYMENT = os.getenv("AZURE_DEPLOYMENT_LLM")

credential = DefaultAzureCredential()
token_provider = get_bearer_token_provider(credential, "https://cognitiveservices.azure.com/.default")

llm = AzureChatOpenAI(
    openai_api_version=OPENAI_API_VERSION,
    azure_deployment=AZURE_DEPLOYMENT,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    azure_ad_token_provider=token_provider,
    temperature=0,
)

# ── Define tools ────────────────────────────────────────────────────

@tool
def get_weather(city: str) -> str:
    """Get the current weather for a given city. Returns a short weather description."""
    # Fake weather data for demonstration
    weather_data = {
        "london": "Cloudy, 12°C, light rain expected",
        "new york": "Sunny, 24°C, clear skies",
        "tokyo": "Partly cloudy, 18°C, humid",
        "paris": "Overcast, 15°C, chance of showers",
        "warsaw": "Snowy, -2°C, heavy snowfall",
    }
    return weather_data.get(city.lower(), f"No weather data available for {city}")


@tool
def multiply(first_number: int, second_number: int) -> int:
    """Multiply two integers together and return the result."""
    return first_number * second_number


@tool
def get_word_length(word: str) -> int:
    """Returns the number of characters in a word."""
    return len(word)


tools = [get_weather, multiply, get_word_length]

# Print tool metadata
print("=== Registered Tools ===")
for t in tools:
    print(f"  {t.name}: {t.description}  args={t.args}")
print()

# ── Create the ReAct agent using LangGraph (modern approach) ────────
# In LangChain 1.x, agents have moved to LangGraph which provides a more
# robust and flexible way to build agent workflows.

SYSTEM_PROMPT = (
    "You are a helpful assistant. "
    "You can use tools to answer questions. "
    "Always explain your reasoning step by step before giving the final answer. "
    "When you use a tool, explain WHY you are using it."
)

# create_react_agent handles the Thought -> Action -> Observation loop automatically
agent_executor = create_react_agent(
    model=llm,
    tools=tools,
    prompt=SYSTEM_PROMPT,
)


def run_agent(query: str) -> str:
    """Run the agent and return the final answer."""
    result = agent_executor.invoke({"messages": [HumanMessage(content=query)]})
    # The result contains all messages; the last one is the final answer
    return result["messages"][-1].content

# ── Run examples ────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("Example 1: Simple tool call (weather)")
print("=" * 60)
result1 = run_agent("What's the weather in Warsaw?")
print(f"\nFinal answer: {result1}\n")

print("\n" + "=" * 60)
print("Example 2: Multi-step reasoning (math + word length)")
print("=" * 60)
result2 = run_agent(
    "How many letters are in the word 'intelligence'? "
    "Multiply that number by 7. What's the result?"
)
print(f"\nFinal answer: {result2}\n")

print("\n" + "=" * 60)
print("Example 3: Agent decides it does NOT need a tool")
print("=" * 60)
result3 = run_agent("What is the capital of France?")
print(f"\nFinal answer: {result3}\n")

print("\n" + "=" * 60)
print("Example 4: Multiple tools in sequence")
print("=" * 60)
result4 = run_agent(
    "I'm planning a trip. What's the weather in London and Tokyo? "
    "Also, if a hotel costs 120 per night and I stay 5 nights, what's the total?"
)
print(f"\nFinal answer: {result4}\n")
