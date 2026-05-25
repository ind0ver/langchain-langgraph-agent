import subprocess
from datetime import datetime
from langchain_core.tools import tool
from ddgs import DDGS
from agent.rag import search_vector_store
from agent.sqlite_db import run_sql_query, get_db_schema
from agent.rag import search_vector_store
from config import MAX_WEB_SEARCH_RESULTS


@tool
def get_current_time() -> str:
    """Get the current date and time in ISO format.
    Useful when the user asks about the current date, time, or when
    something is happening relative to now.
    """
    return datetime.now().isoformat()


@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression and return the result.
    Supports basic arithmetic: +, -, *, /, **, %, parentheses.
    Use this when the user needs any calculation performed.

    Args:
        expression: A mathematical expression like '2 + 2' or '(10 * 5) / 3'
    """
    allowed_chars = set("0123456789+-*/.() %")
    if not all(c in allowed_chars for c in expression.replace(" ", "")):
        return "Error: Expression contains invalid characters."
    try:
        result = eval(expression, {"__builtins__": {}})  # noqa: S307
        return str(result)
    except Exception as e:
        return f"Error evaluating expression: {e}"


@tool
def search_web(query: str) -> str:
    """Search the web for current information about a topic.
    Use this when you need up-to-date facts, news, or data that
    might not be in your training data.

    Args:
        query: The search query string
    """
    with DDGS() as ddgs:
        results = list(ddgs.text(query, max_results=MAX_WEB_SEARCH_RESULTS))
    return "\n".join([f"- {r['title']}: {r['body']}" for r in results])


@tool
def run_terminal_command(command: str) -> str:
    """
    Executes a system terminal command and returns the output. 
    Use this for OS-level actions (like ping, ls, dir).
    The command must be a full string (e.g., "ls -l").
    """
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, check=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        return f"Error executing command: {e.stderr}"
    except Exception as e:
        return f"An unexpected error occurred: {e}"


@tool
def run_powershell_command(command: str) -> str:
    """
    Executes a full command using Windows PowerShell. 
    Use this tool ONLY when the user explicitly requires an OS-level action 
    (e.g., checking network status, listing files, modifying local settings).
    
    WARNING: Using this tool executes code directly on the host operating system. 
    Be careful and ensure the command is safe and necessary.
    
    Args:
        command: The full PowerShell command string (e.g., 'Get-Process | Out-GridView').

    Returns:
        A string containing the standard output (stdout) or an error message.
    """
    print(f"--- Executing PowerShell command: {command} ---")
    try:
        result = subprocess.run(
            ['powershell', '-Command', command], 
            capture_output=True, 
            text=True, 
            check=True, 
            encoding='utf-8'
        )
        
        return result.stdout

    except subprocess.CalledProcessError as e:
        error_output = e.stderr if e.stderr else "Unknown Error"
        return f"Error executing PowerShell command. Exit Code: {e.returncode}\nDetails:\n{error_output}"
    except FileNotFoundError:
        return "Error: PowerShell executable was not found. Please ensure PowerShell is installed and available in the system PATH."
    except Exception as e:
        return f"An unexpected critical error occurred during command execution: {str(e)}"


# Collect all tools into a list for the graph
agent_tools = [get_current_time, calculate, search_web, search_vector_store, run_sql_query, get_db_schema, search_vector_store, run_terminal_command]

sensitive_tools = {"run_terminal_command", "run_powershell_command"}  # Tools that need human approval
