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
print("""
╔═══════════════════════════════════════════════════════════════════╗
║              ReAct Agent Demo — LangChain + LangGraph             ║
╠═══════════════════════════════════════════════════════════════════╣
║  ReAct = Reasoning + Acting                                       ║
║                                                                   ║
║  The loop:   THOUGHT  →  ACTION  →  OBSERVATION  →  (repeat)      ║
║                                                                   ║
║  • THOUGHT:     LLM reasons about what to do next                 ║
║  • ACTION:      LLM picks a tool and provides arguments           ║
║  • OBSERVATION: Tool executes and returns a result                ║
║  • Loop continues until LLM decides it can answer                 ║
╚═══════════════════════════════════════════════════════════════════╝
""")

print("🔧 AVAILABLE TOOLS (the agent can pick from these):")
print("─" * 65)
for t in tools:
    print(f"  • {t.name}")
    print(f"    Description: {t.description}")
    print(f"    Arguments:   {t.args}")
    print()
print("─" * 65)

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


def run_agent(query: str, verbose: bool = True) -> str:
    """
    Run the agent and return the final answer.
    
    When verbose=True, prints each step of the ReAct loop:
      - THOUGHT:      What the LLM is reasoning about
      - ACTION:       Which tool the LLM decides to call (with arguments)
      - OBSERVATION:  The result returned by the tool
      - FINAL ANSWER: The agent's conclusion after all reasoning
    
    This lets students SEE the Thought -> Action -> Observation cycle.
    """
    result = agent_executor.invoke({"messages": [HumanMessage(content=query)]})
    
    if verbose:
        print("\n┌─────────────────────────────────────────────────────────────┐")
        print("│                    ReAct Loop Trace                         │")
        print("└─────────────────────────────────────────────────────────────┘")
        
        step_number = 0
        tools_were_used = False  # Track if any tools were called
        
        for msg in result["messages"]:
            msg_type = type(msg).__name__
            
            # HumanMessage = the original user query
            if msg_type == "HumanMessage":
                print(f"\n📥 USER QUERY:")
                print(f"   \"{msg.content}\"")
            
            # AIMessage = the LLM's response (may include tool calls)
            elif msg_type == "AIMessage":
                # Check if the LLM decided to call tool(s)
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    tools_were_used = True
                    for tc in msg.tool_calls:
                        step_number += 1
                        print(f"\n🔷 STEP {step_number}")
                        print(f"   💭 THOUGHT: The LLM decided it needs to use a tool to answer.")
                        print(f"   🔧 ACTION:  Call tool '{tc['name']}' with args: {tc['args']}")
                else:
                    # No tool call in THIS message
                    if msg.content:
                        if tools_were_used:
                            # This is the final answer AFTER using tools
                            print(f"\n🏁 FINAL REASONING (after tool observations):")
                        else:
                            # No tools were used at all — LLM answered directly
                            print(f"\n💬 LLM RESPONSE (no tools needed — answered from knowledge):")
                        # Truncate very long responses for readability
                        content_preview = msg.content[:500] + "..." if len(msg.content) > 500 else msg.content
                        print(f"   {content_preview}")
            
            # ToolMessage = the result from executing a tool
            elif msg_type == "ToolMessage":
                print(f"   👁️ OBSERVATION: Tool returned → \"{msg.content}\"")
        
        print("\n" + "─" * 65)
    
    # The last message is always the final answer
    return result["messages"][-1].content



# ── Run examples ────────────────────────────────────────────────────
# Each example demonstrates a different aspect of the ReAct pattern.
# Watch the "ReAct Loop Trace" output to see exactly how the agent reasons!

print("\n" + "=" * 65)
print("📚 Example 1: SINGLE TOOL CALL")
print("=" * 65)
print("""
LEARNING POINT: The simplest ReAct scenario.
- Agent receives a question it cannot answer from training data alone.
- It recognizes 'weather' keyword and decides to use get_weather tool.
- After ONE action-observation cycle, it has enough info for the answer.
""")
result1 = run_agent("What's the weather in Warsaw?")
print(f"✅ FINAL ANSWER: {result1}\n")

print("\n" + "=" * 65)
print("📚 Example 2: MULTI-STEP REASONING (chained tool calls)")
print("=" * 65)
print("""
LEARNING POINT: The agent must chain multiple tools in sequence.
- First, it calls get_word_length to count letters.
- Then, it uses that OBSERVATION as input for the multiply tool.
- This shows how the agent "remembers" previous results and reasons over them.
""")
result2 = run_agent(
    "How many letters are in the word 'intelligence'? "
    "Multiply that number by 7. What's the result?"
)
print(f"✅ FINAL ANSWER: {result2}\n")

print("\n" + "=" * 65)
print("📚 Example 3: NO TOOL NEEDED")
print("=" * 65)
print("""
LEARNING POINT: The agent is NOT forced to use tools!
- It evaluates whether a tool would help.
- For general knowledge questions, it relies on its training data.
- This shows the REASONING part of ReAct — deciding when NOT to act.
""")
result3 = run_agent("What is the capital of France?")
print(f"✅ FINAL ANSWER: {result3}\n")

print("\n" + "=" * 65)
print("📚 Example 4: MULTIPLE PARALLEL CONCERNS")
print("=" * 65)
print("""
LEARNING POINT: Complex queries may require multiple different tools.
- The agent breaks down the question into sub-tasks.
- It may call the same tool multiple times (weather for 2 cities).
- It also uses a completely different tool (multiply for cost).
- Watch how it combines all observations into one coherent final answer.
""")
result4 = run_agent(
    "I'm planning a trip. What's the weather in London and Tokyo? "
    "Also, if a hotel costs 120 per night and I stay 5 nights, what's the total?"
)
print(f"✅ FINAL ANSWER: {result4}\n")

# ── Recap for students ──────────────────────────────────────────────
print("""
╔═══════════════════════════════════════════════════════════════════╗
║                         🎓 KEY TAKEAWAYS                          ║
╠═══════════════════════════════════════════════════════════════════╣
║                                                                   ║
║  1. ReAct = Reasoning + Acting                                    ║
║     The LLM doesn't just respond — it THINKS about which tools    ║
║     to use, calls them, observes the results, and reasons again.  ║
║                                                                   ║
║  2. The loop is DYNAMIC                                           ║
║     Unlike a fixed chain, the agent decides at each step:         ║
║       • Do I need a tool? Which one?                              ║
║       • What arguments should I pass?                             ║
║       • Do I have enough info now, or should I continue?          ║
║                                                                   ║
║  3. Tools extend LLM capabilities                                 ║
║     The LLM can't check real weather or do precise math well.     ║
║     Tools give it "superpowers" — external APIs, databases, etc.  ║
║                                                                   ║
║  4. The system prompt matters                                     ║
║     We told the agent to "explain reasoning step by step."        ║
║     This encourages Chain-of-Thought style outputs.               ║
║                                                                   ║
║  5. create_react_agent handles the loop                           ║
║     Under the hood, LangGraph orchestrates the Thought-Action-    ║
║     Observation cycle. You just provide the LLM and tools.        ║
║                                                                   ║
╚═══════════════════════════════════════════════════════════════════╝

Next steps for students:
  • Try adding your own tool (e.g., a calculator, web search, database lookup)
  • Modify the system prompt and see how the agent's behavior changes
  • Ask a question that requires 3+ tool calls in sequence
  • Ask a question where the agent might use the WRONG tool — observe!
""")
