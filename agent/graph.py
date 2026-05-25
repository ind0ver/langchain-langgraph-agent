from langgraph.graph import StateGraph, START, END
from agent.state import AgentState
from agent.nodes import tool_node, reasoning_node, output_node, approval_node


MAX_ITERATIONS = 10


def should_continue(state: AgentState) -> str:
    """Route after the reasoning node.

    Returns:
        'tools' if the LLM wants to call a tool
        'output' if the LLM produced a final answer
        'output' if we hit the iteration safety limit
    """
    last_message = state["messages"][-1]

    # Safety limit: prevent infinite loops
    if state.get("iteration", 0) >= MAX_ITERATIONS:
        return "output"

    # Check if the LLM requested tool calls
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"

    return "output"


def build_agent_graph(checkpointer=None):
    """Construct and compile the agent graph.

    Args:
        checkpointer: Optional checkpointer for state persistence.

    Returns:
        A compiled LangGraph that can be invoked or streamed.
    """
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("reasoning", reasoning_node)
    graph.add_node("tools", tool_node)
    graph.add_node("output", output_node)

    # Define edges
    graph.add_edge(START, "reasoning")
    graph.add_conditional_edges(
        "reasoning",
        should_continue,
        {"tools": "tools", "output": "output"},
    )
    graph.add_edge("tools", "reasoning")  # Loop back after tool execution
    graph.add_edge("output", END)

    # Compile with optional checkpointer
    return graph.compile(checkpointer=checkpointer)


def build_agent_graph_with_hitl(checkpointer=None):
    """Build graph with human-in-the-loop approval."""
    graph = StateGraph(AgentState)

    graph.add_node("reasoning", reasoning_node)
    graph.add_node("approval", approval_node)
    graph.add_node("tools", tool_node)
    graph.add_node("output", output_node)

    graph.add_edge(START, "reasoning")
    graph.add_conditional_edges(
        "reasoning",
        should_continue,
        {"tools": "tools", "approval": "approval", "output": "output"},
    )
    graph.add_edge("approval", "tools")
    graph.add_edge("tools", "reasoning")
    graph.add_edge("output", END)

    return graph.compile(checkpointer=checkpointer)
