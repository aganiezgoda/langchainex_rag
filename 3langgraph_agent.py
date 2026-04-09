# LangGraph Stateful Agent Tutorial
# https://langchain-ai.github.io/langgraph/
# https://langchain-ai.github.io/langgraph/tutorials/introduction/

"""
LangGraph: Graph-Based Agent Workflows
======================================

LangGraph is a library for building stateful, multi-step agent workflows as graphs.
Unlike the ReAct pattern (which is a single loop), LangGraph lets you define:

1. STATE:  A typed dictionary that persists across all steps
2. NODES:  Functions that read/modify state (LLM calls, tool execution, etc.)
3. EDGES:  Connections between nodes (can be conditional)
4. GRAPH:  The compiled workflow that orchestrates everything

Why use LangGraph over AgentExecutor?
- Explicit control over the agent's decision flow
- Better debugging and visualization
- Support for complex workflows (branching, cycles, human-in-the-loop)
- Persistence and streaming built-in

This tutorial builds the same agent as the ReAct example, but using LangGraph.
You'll see how the graph structure gives you more visibility and control.
"""

from typing import TypedDict, Annotated, Sequence
from typing_extensions import Literal
import operator

from langchain_openai import AzureChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from dotenv import load_dotenv
import os

load_dotenv()

# =============================================================================
# STEP 1: AZURE OPENAI SETUP
# =============================================================================

OPENAI_API_VERSION = os.getenv("OPENAI_API_VERSION")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_DEPLOYMENT = os.getenv("AZURE_DEPLOYMENT_LLM")

credential = DefaultAzureCredential()
token_provider = get_bearer_token_provider(
    credential, "https://cognitiveservices.azure.com/.default"
)

llm = AzureChatOpenAI(
    openai_api_version=OPENAI_API_VERSION,
    azure_deployment=AZURE_DEPLOYMENT,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    azure_ad_token_provider=token_provider,
    temperature=0,
)

# =============================================================================
# STEP 2: DEFINE TOOLS (same as ReAct example)
# =============================================================================

@tool
def get_weather(city: str) -> str:
    """Get the current weather for a given city. Returns a short weather description."""
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

# Bind tools to the LLM so it knows what's available
llm_with_tools = llm.bind_tools(tools)

# =============================================================================
# STEP 3: DEFINE THE AGENT STATE
# =============================================================================
# The state is a TypedDict that flows through every node in the graph.
# We use Annotated with operator.add to APPEND new messages rather than replace.

class AgentState(TypedDict):
    """
    The state that persists throughout the agent's execution.
    
    messages: The conversation history. New messages are appended (not replaced)
              thanks to the Annotated[..., operator.add] pattern.
    """
    messages: Annotated[Sequence[BaseMessage], operator.add]


# =============================================================================
# STEP 4: DEFINE THE NODES
# =============================================================================
# Nodes are functions that take the current state and return updates to it.

SYSTEM_PROMPT = SystemMessage(content=(
    "You are a helpful assistant with access to tools. "
    "When a task requires multiple steps where one step depends on the result of another, "
    "call tools SEQUENTIALLY - wait for each result before calling the next tool. "
    "For example, if asked to get a word's length and then multiply it, first call get_word_length, "
    "then use the actual result to call multiply. Do NOT guess values for dependent calculations."
))


def agent_node(state: AgentState) -> dict:
    """
    The 'agent' node: calls the LLM to decide what to do next.
    
    The LLM will either:
      - Return a response with tool_calls (action to take)
      - Return a final response (no tool_calls → we're done)
    """
    print("\n🤖 [Agent Node] Calling LLM...")
    messages = state["messages"]
    # Prepend system prompt if not already present
    if not messages or not isinstance(messages[0], SystemMessage):
        messages = [SYSTEM_PROMPT] + list(messages)
    response = llm_with_tools.invoke(messages)
    print(f"   LLM response: {response.content[:100]}..." if response.content else "   (tool call)")
    return {"messages": [response]}


# For tool execution, we use LangGraph's built-in ToolNode
tool_node = ToolNode(tools)


# =============================================================================
# STEP 5: DEFINE THE ROUTING LOGIC
# =============================================================================
# This function determines which edge to take after the agent node.

def should_continue(state: AgentState) -> Literal["tools", "end"]:
    """
    Decide whether to continue to tools or end the conversation.
    
    - If the last message has tool_calls → go to 'tools' node
    - Otherwise → go to END (we have a final answer)
    """
    last_message = state["messages"][-1]
    
    # Check if the LLM wants to call tools
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        print(f"   → Routing to TOOLS (tools requested: {[tc['name'] for tc in last_message.tool_calls]})")
        return "tools"
    else:
        print("   → Routing to END (final answer ready)")
        return "end"


# =============================================================================
# STEP 6: BUILD THE GRAPH
# =============================================================================
# Now we assemble everything into a StateGraph.

print("=== Building LangGraph Agent ===\n")

# Create the graph with our state schema
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("agent", agent_node)    # The LLM decision-maker
workflow.add_node("tools", tool_node)      # The tool executor

# Set the entry point (where the graph starts)
workflow.set_entry_point("agent")

# Add edges
# 1. Conditional edge from agent → either "tools" or END
workflow.add_conditional_edges(
    "agent",           # From this node...
    should_continue,   # Use this function to decide...
    {
        "tools": "tools",  # If "tools" → go to tools node
        "end": END,        # If "end" → finish
    }
)

# 2. After tools execute, always go back to agent (to process results)
workflow.add_edge("tools", "agent")

# Compile the graph into a runnable
graph = workflow.compile()

# =============================================================================
# STEP 7: VISUALIZE THE GRAPH (optional)
# =============================================================================
# LangGraph can generate a Mermaid diagram of your workflow!

print("Graph structure (Mermaid):")
print("-" * 40)
try:
    print(graph.get_graph().draw_mermaid())
except Exception as e:
    print(f"(Visualization requires additional dependencies: {e})")
print("-" * 40)


# =============================================================================
# STEP 8: HELPER FUNCTION TO RUN THE AGENT
# =============================================================================

def run_agent(question: str) -> str:
    """
    Run the agent with a question and return the final answer.
    
    This invokes the compiled graph and streams through each step.
    """
    print(f"\n📝 Question: {question}")
    print("=" * 60)
    
    # Create the initial state with the user's question
    initial_state = {"messages": [HumanMessage(content=question)]}
    
    # Run the graph
    final_state = graph.invoke(initial_state)
    
    # Extract the final answer from the last message
    final_message = final_state["messages"][-1]
    return final_message.content


# =============================================================================
# STEP 9: RUN EXAMPLES
# =============================================================================

if __name__ == "__main__":
    
    print("\n" + "=" * 60)
    print("Example 1: Simple tool call (weather)")
    print("=" * 60)
    answer1 = run_agent("What's the weather in Warsaw?")
    print(f"\n✅ Final answer: {answer1}\n")

    print("\n" + "=" * 60)
    print("Example 2: Multi-step reasoning (math + word length)")
    print("=" * 60)
    answer2 = run_agent(
        "How many letters are in the word 'intelligence'? "
        "Multiply that number by 7. What's the result?"
    )
    print(f"\n✅ Final answer: {answer2}\n")

    print("\n" + "=" * 60)
    print("Example 3: Agent decides it does NOT need a tool")
    print("=" * 60)
    answer3 = run_agent("What is the capital of France?")
    print(f"\n✅ Final answer: {answer3}\n")

    print("\n" + "=" * 60)
    print("Example 4: Multiple tools in sequence")
    print("=" * 60)
    answer4 = run_agent(
        "I'm planning a trip. What's the weather in London and Tokyo? "
        "Also, if a hotel costs 120 per night and I stay 5 nights, what's the total?"
    )
    print(f"\n✅ Final answer: {answer4}\n")

    # ==========================================================================
    # BONUS: Inspecting the full message history
    # ==========================================================================
    print("\n" + "=" * 60)
    print("BONUS: Full message history from last run")
    print("=" * 60)
    
    # Re-run to capture full state
    final_state = graph.invoke({
        "messages": [HumanMessage(content="What's 7 times 8?")]
    })
    
    print("\nMessage trace:")
    for i, msg in enumerate(final_state["messages"]):
        msg_type = type(msg).__name__
        content = str(msg.content)[:80] if msg.content else "(no content)"
        print(f"  [{i}] {msg_type}: {content}")
