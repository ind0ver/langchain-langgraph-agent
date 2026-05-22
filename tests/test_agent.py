"""Integration tests for the agent graph."""
import pytest
from langchain_core.messages import HumanMessage, AIMessage
from agent.graph import build_agent_graph
from agent.state import AgentState
from agent.tools import calculate, get_current_time


def test_graph_compiles():
    """The graph should compile without errors."""
    graph = build_agent_graph()
    assert graph is not None


def test_calculate_tool():
    """The calculate tool should return correct results."""
    assert calculate.invoke({"expression": "2 + 2"}) == "4"
    assert calculate.invoke({"expression": "10 * 5"}) == "50"
    assert calculate.invoke({"expression": "(100 - 25) / 3"}) == "25.0"


def test_calculate_tool_rejects_invalid():
    """The calculate tool should reject dangerous input."""
    result = calculate.invoke({"expression": "__import__('os').system('ls')"})
    assert "Error" in result or "invalid" in result


def test_time_tool_returns_iso():
    """The time tool should return an ISO formatted string."""
    result = get_current_time.invoke({})
    assert "2026" in result
    assert "T" in result  # ISO format separator


def test_state_schema():
    """State should accept valid field types."""
    state: AgentState = {
        "messages": [HumanMessage(content="test")],
        "tool_calls_count": 0,
        "final_answer": "",
        "iteration": 0,
    }
    assert len(state["messages"]) == 1
    assert state["iteration"] == 0


def test_agent_responds(monkeypatch):
    """The agent should produce a final answer (requires API key)."""
    graph = build_agent_graph()
    result = graph.invoke({
        "messages": [HumanMessage(content="What is 15 + 27?")],
        "tool_calls_count": 0,
        "final_answer": "",
        "iteration": 0,
    })
    assert result["final_answer"] != ""
    assert result["iteration"] >= 1


# Run with:
# pytest tests/ -v
