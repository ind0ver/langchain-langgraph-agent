import sqlite_utils
from langchain_core.tools import tool
from config import DB_PATH


@tool
def run_sql_query(query: str) -> str:
    """Execute an SQL query against the Chinook database.
    The database contains tables: tracks, albums, artists, customers, invoices.
    Use this tool for any data-related questions, for example,
    'top 5 longest tracks', 'number of customers in Canada'.
    """
    db = sqlite_utils.Database(DB_PATH)
    try:
        results = list(db.query(query))
        if not results:
            return "Query executed but returned no data."
        # Limit output to avoid overloading the LLM context
        return str(results[:50])
    except Exception as e:
        return f"Error executing query: {e}. Please check the syntax."

@tool
def get_db_schema() -> str:
    """Returns the schema of the Chinook database.
    Call this tool first whenever you need any information from the DB,
    to understand which tables and columns are available.
    """
    db = sqlite_utils.Database(DB_PATH)
    return str(db.schema)