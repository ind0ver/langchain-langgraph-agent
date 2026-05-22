"""Agent state schema with typed fields and reducers."""
from typing import Annotated, TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """State that flows through the agent graph.

    Attributes:
        messages: Conversation history with reducer that appends new messages.
        tool_calls_count: Tracks how many tools the agent has called.
        final_answer: The agent's synthesized response to the user.
        iteration: Current reasoning loop iteration (for safety limits).
    """
    messages: Annotated[list[BaseMessage], add_messages]
    tool_calls_count: int
    final_answer: str
    iteration: int
