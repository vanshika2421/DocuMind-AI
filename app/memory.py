
import sqlite3
from pathlib import Path


# Database location
DB_PATH = Path(__file__).resolve().parent.parent / "documind.db"


# --------------------------------------------------
# DATABASE CONNECTION
# --------------------------------------------------

def get_connection():
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    return connection


# --------------------------------------------------
# CHAT MEMORY
# --------------------------------------------------

def initialize_memory():
    """Create chat history table and index."""

    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS chat_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_chat_session
            ON chat_history(session_id, id)
        """)


def get_history(session_id: str, limit: int = 10):
    """Retrieve the latest chat messages for a session."""

    with get_connection() as conn:
        rows = conn.execute("""
            SELECT role, content
            FROM chat_history
            WHERE session_id = ?
            ORDER BY id DESC
            LIMIT ?
        """, (session_id, limit)).fetchall()

    return [
        {
            "role": row["role"],
            "content": row["content"]
        }
        for row in reversed(rows)
    ]


def save_turn(session_id: str, question: str, answer: str):
    """Save one user question and assistant answer."""

    with get_connection() as conn:
        conn.executemany("""
            INSERT INTO chat_history (session_id, role, content)
            VALUES (?, ?, ?)
        """, [
            (session_id, "user", question),
            (session_id, "assistant", answer)
        ])


def clear_history(session_id: str):
    """Delete all chat messages for a session."""

    with get_connection() as conn:
        conn.execute("""
            DELETE FROM chat_history
            WHERE session_id = ?
        """, (session_id,))


# --------------------------------------------------
# DOCUMENT MANAGEMENT
# --------------------------------------------------

def initialize_documents():
    """Create document registry table."""

    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                document_id TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                chunk_count INTEGER DEFAULT 0,
                status TEXT DEFAULT 'indexed',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)


def save_document(
    document_id: str,
    filename: str,
    chunk_count: int
):
    """Save document metadata after successful ingestion."""

    with get_connection() as conn:
        conn.execute("""
            INSERT OR REPLACE INTO documents (
                document_id,
                filename,
                chunk_count,
                status
            )
            VALUES (?, ?, ?, 'indexed')
        """, (
            document_id,
            filename,
            chunk_count
        ))


def list_documents():
    """Return all registered documents."""

    with get_connection() as conn:
        rows = conn.execute("""
            SELECT
                document_id,
                filename,
                chunk_count,
                status,
                created_at
            FROM documents
            ORDER BY created_at DESC
        """).fetchall()

    return [dict(row) for row in rows]


def remove_document_record(document_id: str):
    """Remove a document from the SQLite registry."""

    with get_connection() as conn:
        conn.execute("""
            DELETE FROM documents
            WHERE document_id = ?
        """, (document_id,))


# --------------------------------------------------
# INITIALIZE ALL DATABASE TABLES
# --------------------------------------------------

def initialize_database():
    """Initialize all DocuMind AI database tables."""

    initialize_memory()
    initialize_documents()
