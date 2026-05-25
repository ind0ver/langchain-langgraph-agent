"""Complete LangGraph Agent — Full working example."""
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from agent.graph import build_agent_graph


def main():
    """Run the complete agent with memory and streaming."""
    with SqliteSaver.from_conn_string("checkpoints.db") as checkpointer:
        graph = build_agent_graph(checkpointer=checkpointer)
        config = {"configurable": {"thread_id": "demo-session"}}

        queries = [
            "What is the square root of 144 plus 25?",  # Tests calculator tool
            "Search the web for the latest Python release.",  # Tests web search
            # "What where my previous requests and what tools did you use to answer them?",  # Tests memory
        ]

        for query in queries:
            print(f"\nUser: {query}")
            print("-" * 50)

            for event in graph.stream(
                {"messages": [HumanMessage(content=query)],
                 "tool_calls_count": 0, "final_answer": "", "iteration": 0},
                config=config,
                stream_mode="updates",
            ):
                for node_name, node_output in event.items():
                    print(f"[Node: {node_name}]")
                    if node_name == "reasoning":
                        msg = node_output["messages"][-1]
                        if hasattr(msg, "tool_calls") and msg.tool_calls:
                            for tc in msg.tool_calls:
                                print(f"  → Calling tool: {tc['name']}({tc['args']})")
                        else:
                            print(f"  → Response: {msg.content[:250]}...")
                    elif node_name == "tools":
                        for msg in node_output.get("messages", []):
                            print(f"  → Tool result: {msg.content[:250]}...")
                    elif node_name == "output":
                        print(f"  → Agent: \n{node_output.get('final_answer', '')}")


if __name__ == "__main__":
    main()