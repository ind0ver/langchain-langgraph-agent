from langchain_ollama import ChatOllama
from agent.tools import agent_tools
from config import MODEL, BASE_URL


llm = ChatOllama(
    model=MODEL,
    base_url=BASE_URL,
    temperature=0.2,
    top_p=0.95,
    top_k=64,
).bind_tools(agent_tools)