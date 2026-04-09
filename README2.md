# ReAct Agent Tutorial with LangChain & Azure OpenAI

This tutorial demonstrates how to build a **ReAct (Reasoning + Acting) agent** using LangChain with Azure OpenAI.

## What is ReAct?

ReAct is an agent pattern where the LLM follows an iterative loop:

```
Thought → Action → Observation → Thought → ... → Final Answer
```

| Step | Description |
|------|-------------|
| **Thought** | The LLM reasons about the current state and decides what to do next |
| **Action** | The LLM picks a tool and provides arguments for it |
| **Observation** | The tool is executed and returns a result |
| **Repeat** | The loop continues until the LLM has enough info to produce a final answer |

### How is this different from a simple chain?

- The agent decides **which tools** to use and **how many times**
- The agent can reason about intermediate results before deciding next steps
- There is no fixed sequence — the flow is **dynamic** based on the problem

## Tools in This Example

| Tool | Description | Input | Output |
|------|-------------|-------|--------|
| `get_weather` | Returns weather for a city | `city: str` | Weather description |
| `multiply` | Multiplies two numbers | `first_number: int, second_number: int` | Product |
| `get_word_length` | Counts characters in a word | `word: str` | Length |

## Prerequisites

1. **Azure OpenAI Resource** with a deployed chat model (e.g., GPT-4)
2. **Azure CLI** logged in (`az login`)
3. **Environment variables** in `.env`:

```env
OPENAI_API_VERSION=2024-02-15-preview
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_DEPLOYMENT_LLM=gpt-4
```

## Installation

```bash
# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

## Running the Script

```bash
python 2langchain_react_agent.py
```

## Examples Demonstrated

### Example 1: Simple Tool Call
```
Input: "What's the weather in Warsaw?"
→ Agent calls get_weather("warsaw")
→ Returns: "Snowy, -2°C, heavy snowfall"
```

### Example 2: Multi-Step Reasoning
```
Input: "How many letters are in 'intelligence'? Multiply that by 7."
→ Agent calls get_word_length("intelligence") → 12
→ Agent calls multiply(12, 7) → 84
→ Returns: "The word 'intelligence' has 12 letters. 12 × 7 = 84"
```

### Example 3: No Tool Needed
```
Input: "What is the capital of France?"
→ Agent uses its knowledge (no tool call)
→ Returns: "Paris"
```

### Example 4: Multiple Tools in Sequence
```
Input: "What's the weather in London and Tokyo? If a hotel costs 120/night for 5 nights, what's the total?"
→ Agent calls get_weather("london") and get_weather("tokyo")
→ Agent calls multiply(120, 5) → 600
→ Returns combined answer
```

## Key Concepts

### Tool-Calling Agent vs Classic ReAct

This tutorial uses the **tool-calling agent** (`create_tool_calling_agent`), which is the modern approach for chat models like GPT-4. The classic ReAct pattern using text-based prompts is shown in comments for reference.

```python
# Modern approach (recommended for chat models)
agent = create_tool_calling_agent(llm, tools, prompt)

# Classic approach (for completion models)
# react_prompt = hub.pull("hwchase17/react")
# agent = create_react_agent(llm, tools, react_prompt)
```

### AgentExecutor Configuration

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `verbose` | `True` | Prints each step of the reasoning loop |
| `max_iterations` | `10` | Safety limit to prevent infinite loops |
| `handle_parsing_errors` | `True` | Gracefully handles malformed tool calls |
| `return_intermediate_steps` | `True` | Allows inspection of each step for debugging |

### Inspecting Intermediate Steps

With `return_intermediate_steps=True`, you can see exactly what the agent did:

```python
result = agent_executor.invoke({"input": "..."})
print(result['output'])              # Final answer
print(result['intermediate_steps'])  # List of (AgentAction, observation) tuples
```

## References

- [LangChain Agents Concepts](https://python.langchain.com/docs/concepts/agents/#react-agents)
- [Migrating from Legacy Agents](https://python.langchain.com/docs/how_to/migrate_agent/)
- [ReAct Paper](https://arxiv.org/abs/2210.03629)
