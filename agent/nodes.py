"""Node functions for the agent graph."""
from langchain_core.messages import SystemMessage
from langgraph.prebuilt import ToolNode
from langgraph.types import interrupt, Command

from agent.state import AgentState
from agent.tools import agent_tools, sensitive_tools
from agent.llm import llm


SYSTEM_PROMPT = """You are a helpful AI assistant with access to tools.
Use tools when you need current information, calculations, or web data.
Always explain your reasoning before and after using tools.
If you cannot find an answer, say so honestly rather than making one up."""


def reasoning_node(state: AgentState) -> dict:
    """Call the LLM to reason about the next step.

    Reads the current messages and decides whether to:
    1. Call a tool for more information
    2. Provide a final answer to the user
    """
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    response = llm.invoke(messages)
    return {
        "messages": [response],
        "iteration": state.get("iteration", 0) + 1,
    }


# Use LangGraph's built-in ToolNode for automatic tool execution
tool_node = ToolNode(agent_tools)


def output_node(state: AgentState) -> dict:
    """Extract the final answer from the last AI message."""
    last_message = state["messages"][-1]
    return {
        "final_answer": last_message.content,
    }


def approval_node(state: AgentState) -> dict | Command:
    """Pause execution and ask the human to approve tool calls."""
    last_message = state["messages"][-1]
    tool_calls = last_message.tool_calls if hasattr(last_message, "tool_calls") else []

    sensitive_calls = [tc for tc in tool_calls if tc["name"] in sensitive_tools]

    if sensitive_calls:
        # interrupt() pauses the graph and returns this to the caller
        decision = interrupt({
            "question": "The agent wants to search the web. Approve?",
            "tool_calls": [
                {"name": tc["name"], "args": tc["args"]}
                for tc in sensitive_calls
            ],
        })
        if decision.get("approved"):
            return {}  # Continue to tool execution
        else:
            # Skip tool execution, go back to reasoning with feedback
            return Command(
                goto="reasoning",
                update={"messages": [{"role": "user", "content": "Tool call denied by user."}]},
            )
    return {}  # No sensitive tools, proceed normally