# LangGraph Agent with Tools, RAG, SQL, and HITL

A production-ready AI agent built with **LangGraph**, **Ollama** (OpenAI-compatible), and **LangChain**.  
The agent can:
- Use **tools** (web search, SQL queries, calculator, terminal commands)
- Query a **RAG** knowledge base (PDF documents)
- Execute **SQL** on a Chinook database
- Support **Human-in-the-Loop (HITL)** for sensitive operations
- Remember conversation history via **checkpointing**

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🧠 **Reasoning** | LLM decides when to call tools or answer directly |
| 🔧 **Tool Use** | Calculator, web search (DuckDuckGo), system commands, SQL, RAG |
| 📚 **RAG** | Load PDFs → vector store (ChromaDB) → semantic search |
| 🗄️ **SQL Database** | Query Chinook SQLite with schema introspection |
| 🔐 **HITL** | Ask user before running sensitive commands (terminal, PowerShell) |
| 💾 **Memory** | Persistent checkpoints using SQLite |
| 🚦 **Safety Limits** | Max iterations to prevent infinite loops |
| 🌐 **OpenAI Compatible** | Works with Ollama, LocalAI, or any OpenAI API |

---

## 📦 Installation

### Clone the repository
```bash
git clone https://github.com/yourusername/langgraph-agent.git
cd langgraph-agent
```

### Create virtual environment (optional)
```bash
venv\Scripts\activate
```

### Install dependencies
```bash
pip install -r requirements.txt
```

---

## 🔧 Configuration
### Edit config.py:

```bash
### LLM (OpenAI-compatible endpoint)
BASE_URL = "http://localhost:11434/v1"   # Ollama's /v1 endpoint
MODEL = "gemma2:9b"                      # or any Ollama model

### SQLite database
DB_PATH = "chinook_db/Chinook_Sqlite.sqlite"

### RAG settings
DOCS_DIRECTORY = "./rag_docs"            # place PDFs here
PERSIST_DIRECTORY = "./chroma_db"        # vector store cache
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Web search
MAX_WEB_SEARCH_RESULTS = 5
```

---

## 🚀 Usage

### 1. Make sure the BASE_URL from config.py is an OpenAI-compatible endpoint (e.g. a running Ollama)

### 2. Prepare data (optional)

- **RAG documents:** Put PDF files into `rag_docs/`.
- **Chinook database:** Download from [here](https://github.com/lerocha/chinook-database) and place at `chinook_db/Chinook_Sqlite.sqlite`.

### 3. Run the agent

```bash
python main.py
```

The agent will process a few example queries.

---

## 🧠 Graph Architecture

#### The agent uses a `StateGraph` with the following nodes:

<img width="2114" height="577" alt="deepseek_mermaid_20260525_772950" src="https://github.com/user-attachments/assets/b1621ca2-d734-4a67-93fa-7eb7c616b12a" />

#### With `HITL` enabled, sensitive tools trigger an approval node:

<img width="2223" height="834" alt="deepseek_mermaid_20260525_43286b" src="https://github.com/user-attachments/assets/0423f06c-b35d-437d-a872-417f726066fa" />

---

## 🛠️ Available Tools

| Tool | Description | Sensitive? |
|------|-------------|------------|
| `get_current_time` | Returns current datetime | ❌ |
| `calculate` | Safe math expression evaluator | ❌ |
| `search_web` | DuckDuckGo search | ❌ |
| `search_vector_store` | RAG query on your PDFs | ❌ |
| `run_sql_query` | Execute SQL queries on Chinook DB | ❌ |
| `get_db_schema` | Show SQL table/column info | ❌ |
| `run_terminal_command` | Run any system command | ✅ |
| `run_powershell_command` | Run PowerShell command (Windows) | ✅ |

Sensitive tools require human approval when HITL is enabled.

---

## 🔐 Human-in-the-Loop (HITL)

To enable approval for sensitive tools, use the HITL graph:

```python
from agent.graph import build_agent_graph_with_hitl

graph = build_agent_graph_with_hitl(checkpointer=checkpointer)
```

When a sensitive tool is called, execution pauses and the user is asked:

```
The agent wants to use the next tool. Approve?
Tool: run_terminal_command, args: {'command': 'rm -rf /tmp/test'}
```

The user can respond with `{"approved": True}` or `{"approved": False}`.

HITL is implemented using `interrupt()` from LangGraph – the graph resumes automatically after receiving input.

---

## 🔄 Extending the Agent

### Add a new tool

1. Define a function with `@tool` decorator in `tools.py`.
2. Add it to the `agent_tools` list.
3. *(Optional)* Add its name to `sensitive_tools` if it needs approval.

**Example:**

```python
@tool
def get_weather(city: str) -> str:
    """Get current weather for a city."""
    # your API call here
    return f"Weather in {city}: sunny, 22°C"
```

---
