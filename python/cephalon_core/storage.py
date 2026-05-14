import datetime as dt
import json
import os
import sqlite3
import shutil
import threading
import time
from typing import Any

import lancedb
import pyarrow as pa

from .config import ACTIVE_VECTOR_TABLE, EMBEDDING_DIMENSION, EMBEDDING_MODEL_ID, Settings
from .schemas import RagSettings


SQLITE_LOCK = threading.RLock()

VECTOR_SCHEMA = pa.schema([
    pa.field("vector", pa.list_(pa.float32(), EMBEDDING_DIMENSION)),
    pa.field("id", pa.string()),
    pa.field("doc_id", pa.string()),
    pa.field("text", pa.string()),
    pa.field("chunk_index", pa.int64()),
    pa.field("parent_id", pa.string()),
    pa.field("source_kind", pa.string()),
    pa.field("embedding_model_id", pa.string()),
    pa.field("embedding_dim", pa.int64()),
    pa.field("chunk_length", pa.int64()),
])


def vector_schema(dimension: int = EMBEDDING_DIMENSION) -> pa.Schema:
    return pa.schema([
        pa.field("vector", pa.list_(pa.float32(), dimension)),
        pa.field("id", pa.string()),
        pa.field("doc_id", pa.string()),
        pa.field("text", pa.string()),
        pa.field("chunk_index", pa.int64()),
        pa.field("parent_id", pa.string()),
        pa.field("source_kind", pa.string()),
        pa.field("embedding_model_id", pa.string()),
        pa.field("embedding_dim", pa.int64()),
        pa.field("chunk_length", pa.int64()),
    ])


def connect_sqlite(settings: Settings) -> sqlite3.Connection:
    os.makedirs(settings.data_dir, exist_ok=True)
    conn = sqlite3.connect(os.path.join(settings.data_dir, "meta.db"), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    run_migrations(conn, settings)
    return conn


def connect_lance(settings: Settings):
    os.makedirs(settings.data_dir, exist_ok=True)
    return lancedb.connect(os.path.join(settings.data_dir, "lancedb"))


def fetchone(conn: sqlite3.Connection, query: str, params: tuple = ()):
    with SQLITE_LOCK:
        cursor = conn.cursor()
        cursor.execute(query, params)
        return cursor.fetchone()


def fetchall(conn: sqlite3.Connection, query: str, params: tuple = ()):
    with SQLITE_LOCK:
        cursor = conn.cursor()
        cursor.execute(query, params)
        return cursor.fetchall()


def execute(conn: sqlite3.Connection, query: str, params: tuple = (), commit: bool = True):
    with SQLITE_LOCK:
        cursor = conn.cursor()
        cursor.execute(query, params)
        if commit:
            conn.commit()
        return cursor


def executescript(conn: sqlite3.Connection, script: str):
    with SQLITE_LOCK:
        conn.executescript(script)
        conn.commit()


def table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {row["name"] for row in fetchall(conn, f"PRAGMA table_info({table})")}


def add_column_if_missing(conn: sqlite3.Connection, table: str, column: str, definition: str) -> None:
    if column not in table_columns(conn, table):
        execute(conn, f"ALTER TABLE {table} ADD COLUMN {column} {definition}")


def migration_applied(conn: sqlite3.Connection, version: str) -> bool:
    row = fetchone(conn, "SELECT version FROM schema_migrations WHERE version = ?", (version,))
    return row is not None


def mark_migration(conn: sqlite3.Connection, version: str) -> None:
    execute(conn, "INSERT OR IGNORE INTO schema_migrations (version, applied_at) VALUES (?, ?)", (version, int(time.time())))


def run_migrations(conn: sqlite3.Connection, settings: Settings) -> None:
    executescript(conn, """
        CREATE TABLE IF NOT EXISTS schema_migrations (
            version TEXT PRIMARY KEY,
            applied_at INTEGER NOT NULL
        );
    """)

    if not migration_applied(conn, "001_base"):
        executescript(conn, """
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                path TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                ingested_at INTEGER,
                chunk_count INTEGER,
                status TEXT DEFAULT 'pending',
                type TEXT DEFAULT 'file'
            );
            CREATE TABLE IF NOT EXISTS chunks (
                id TEXT PRIMARY KEY,
                doc_id TEXT REFERENCES documents(id) ON DELETE CASCADE,
                chunk_index INTEGER,
                text TEXT NOT NULL
            );
        """)
        mark_migration(conn, "001_base")

    if not migration_applied(conn, "002_workbench"):
        for column, definition in [
            ("display_name", "TEXT"),
            ("size_bytes", "INTEGER DEFAULT 0"),
            ("modified_at", "INTEGER"),
            ("last_error", "TEXT"),
            ("last_indexed_at", "INTEGER"),
        ]:
            add_column_if_missing(conn, "documents", column, definition)

        executescript(conn, """
            CREATE TABLE IF NOT EXISTS document_tags (
                doc_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                tag TEXT NOT NULL,
                PRIMARY KEY (doc_id, tag)
            );
            CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY,
                kind TEXT NOT NULL,
                path TEXT NOT NULL,
                status TEXT NOT NULL,
                total_files INTEGER DEFAULT 0,
                processed_files INTEGER DEFAULT 0,
                skipped_files INTEGER DEFAULT 0,
                current_file TEXT,
                error TEXT,
                created_at INTEGER NOT NULL,
                updated_at INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS job_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                job_id TEXT,
                event_type TEXT NOT NULL,
                payload TEXT NOT NULL,
                created_at INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS app_settings (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
        """)
        execute(conn, "UPDATE documents SET display_name = COALESCE(display_name, path)")
        mark_migration(conn, "002_workbench")

    if not migration_applied(conn, "003_embedding_metadata"):
        for table, columns in {
            "documents": [
                ("embedding_model_id", "TEXT"),
                ("embedding_dim", "INTEGER"),
                ("stale_embedding", "INTEGER DEFAULT 0"),
                ("extraction_mode", "TEXT DEFAULT 'native'"),
                ("last_retrieved_at", "INTEGER"),
                ("retrieval_count", "INTEGER DEFAULT 0"),
            ],
            "chunks": [
                ("chunk_length", "INTEGER DEFAULT 0"),
                ("embedding_model_id", "TEXT"),
                ("embedding_dim", "INTEGER"),
            ],
            "jobs": [
                ("target_doc_id", "TEXT"),
                ("force_text", "INTEGER DEFAULT 0"),
            ],
        }.items():
            for column, definition in columns:
                add_column_if_missing(conn, table, column, definition)
        execute(
            conn,
            "UPDATE documents SET stale_embedding = 1 WHERE type = 'file' AND COALESCE(embedding_dim, 768) != ?",
            (EMBEDDING_DIMENSION,),
        )
        mark_migration(conn, "003_embedding_metadata")

    if not migration_applied(conn, "004_sqlite_fts"):
        ensure_chunks_fts(conn)
        rebuild_chunks_fts(conn)
        mark_migration(conn, "004_sqlite_fts")

    if not migration_applied(conn, "005_hierarchical_chunks"):
        for column, definition in [
            ("parent_id", "TEXT"),
            ("summary_id", "TEXT"),
            ("token_count", "INTEGER DEFAULT 0"),
            ("semantic_role", "TEXT DEFAULT 'child'"),
        ]:
            add_column_if_missing(conn, "chunks", column, definition)
        executescript(conn, """
            CREATE TABLE IF NOT EXISTS parent_chunks (
                id TEXT PRIMARY KEY,
                doc_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                parent_index INTEGER NOT NULL,
                text TEXT NOT NULL,
                summary TEXT NOT NULL,
                token_count INTEGER DEFAULT 0,
                created_at INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS summary_nodes (
                id TEXT PRIMARY KEY,
                doc_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                parent_id TEXT NOT NULL REFERENCES parent_chunks(id) ON DELETE CASCADE,
                chunk_index INTEGER NOT NULL,
                summary TEXT NOT NULL,
                token_count INTEGER DEFAULT 0,
                created_at INTEGER NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_parent_chunks_doc ON parent_chunks(doc_id);
            CREATE INDEX IF NOT EXISTS idx_summary_nodes_doc ON summary_nodes(doc_id);
        """)
        mark_migration(conn, "005_hierarchical_chunks")

    if not migration_applied(conn, "006_conversation_history"):
        executescript(conn, """
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                created_at INTEGER NOT NULL,
                updated_at INTEGER NOT NULL,
                archived INTEGER DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS messages (
                id TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                model TEXT,
                settings_json TEXT,
                meta_json TEXT,
                created_at INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS message_sources (
                message_id TEXT NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
                source_rank INTEGER NOT NULL,
                source_json TEXT NOT NULL,
                PRIMARY KEY (message_id, source_rank)
            );
            CREATE INDEX IF NOT EXISTS idx_messages_conversation ON messages(conversation_id, created_at);
            CREATE INDEX IF NOT EXISTS idx_conversations_updated ON conversations(updated_at DESC);
        """)
        mark_migration(conn, "006_conversation_history")

    execute(
        conn,
        "INSERT OR IGNORE INTO documents (id, path, display_name, content_hash, chunk_count, status, type) VALUES (?, ?, ?, ?, ?, ?, ?)",
        ("core_memory", "Internal AI Memory", "Internal AI Memory", "none", 0, "ready", "memory"),
    )
    ensure_default_settings(conn, settings)


def ensure_chunks_fts(conn: sqlite3.Connection) -> None:
    executescript(conn, """
        CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
            chunk_id UNINDEXED,
            doc_id UNINDEXED,
            text,
            tokenize = 'unicode61'
        );
    """)


def rebuild_chunks_fts(conn: sqlite3.Connection) -> None:
    ensure_chunks_fts(conn)
    with SQLITE_LOCK:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM chunks_fts")
        cursor.execute(
            """
            INSERT INTO chunks_fts (chunk_id, doc_id, text)
            SELECT id, doc_id, text FROM chunks
            """
        )
        conn.commit()


def upsert_chunk_fts(conn: sqlite3.Connection, chunk_id: str, doc_id: str, text: str) -> None:
    ensure_chunks_fts(conn)
    with SQLITE_LOCK:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM chunks_fts WHERE chunk_id = ?", (chunk_id,))
        cursor.execute("INSERT INTO chunks_fts (chunk_id, doc_id, text) VALUES (?, ?, ?)", (chunk_id, doc_id, text))
        conn.commit()


def delete_document_fts(conn: sqlite3.Connection, doc_id: str) -> None:
    ensure_chunks_fts(conn)
    execute(conn, "DELETE FROM chunks_fts WHERE doc_id = ?", (doc_id,))


def delete_document_hierarchy(conn: sqlite3.Connection, doc_id: str) -> None:
    with SQLITE_LOCK:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM summary_nodes WHERE doc_id = ?", (doc_id,))
        cursor.execute("DELETE FROM parent_chunks WHERE doc_id = ?", (doc_id,))
        conn.commit()


def clean_generated_vector_state(settings: Settings, lance_conn, active_table: str = ACTIVE_VECTOR_TABLE) -> str | None:
    """Back up generated vector data before dropping inactive LanceDB tables."""
    try:
        table_names = list(lance_conn.table_names())
    except Exception:
        return None
    stale_tables = [name for name in table_names if name != active_table]
    if not stale_tables:
        return None

    backup_root = os.path.abspath(os.path.expanduser("~/cephalon-data-backups"))
    timestamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    backup_dir = os.path.join(backup_root, f"generated-indexes-{timestamp}")
    os.makedirs(backup_dir, exist_ok=True)
    lance_path = os.path.join(settings.data_dir, "lancedb")
    if os.path.exists(lance_path):
        shutil.copytree(lance_path, os.path.join(backup_dir, "lancedb"), dirs_exist_ok=True)

    for table_name in stale_tables:
        try:
            lance_conn.drop_table(table_name)
        except Exception:
            continue
    return backup_dir


def ensure_default_settings(conn: sqlite3.Connection, settings: Settings) -> None:
    existing = fetchone(conn, "SELECT value FROM app_settings WHERE key = 'rag'")
    if existing:
        return
    defaults = RagSettings(**settings.rag_defaults.__dict__)
    execute(conn, "INSERT INTO app_settings (key, value) VALUES (?, ?)", ("rag", defaults.model_dump_json()))


def get_rag_settings(conn: sqlite3.Connection) -> RagSettings:
    row = fetchone(conn, "SELECT value FROM app_settings WHERE key = 'rag'")
    if not row:
        return RagSettings()
    return RagSettings(**json.loads(row["value"]))


def save_rag_settings(conn: sqlite3.Connection, rag_settings: RagSettings) -> RagSettings:
    execute(
        conn,
        "INSERT INTO app_settings (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        ("rag", rag_settings.model_dump_json()),
    )
    return rag_settings


def row_to_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
    if row is None:
        return None
    return {key: row[key] for key in row.keys()}


def get_document_tags(conn: sqlite3.Connection, doc_id: str) -> list[str]:
    rows = fetchall(conn, "SELECT tag FROM document_tags WHERE doc_id = ? ORDER BY tag", (doc_id,))
    return [row["tag"] for row in rows]


def document_payload(conn: sqlite3.Connection, row: sqlite3.Row) -> dict[str, Any]:
    display_name = row["display_name"] or os.path.basename(row["path"])
    return {
        "id": row["id"],
        "name": display_name if display_name != row["path"] else os.path.basename(row["path"]),
        "path": row["path"],
        "status": row["status"],
        "chunks": row["chunk_count"] or 0,
        "type": row["type"],
        "size_bytes": row["size_bytes"] or 0,
        "modified_at": row["modified_at"],
        "last_error": row["last_error"],
        "last_indexed_at": row["last_indexed_at"],
        "embedding_model_id": row["embedding_model_id"] if "embedding_model_id" in row.keys() else None,
        "embedding_dim": row["embedding_dim"] if "embedding_dim" in row.keys() else None,
        "stale_embedding": bool(row["stale_embedding"]) if "stale_embedding" in row.keys() else False,
        "extraction_mode": row["extraction_mode"] if "extraction_mode" in row.keys() else None,
        "last_retrieved_at": row["last_retrieved_at"] if "last_retrieved_at" in row.keys() else None,
        "retrieval_count": row["retrieval_count"] if "retrieval_count" in row.keys() else 0,
        "tags": get_document_tags(conn, row["id"]),
    }


def active_vector_table_name(app_state=None) -> str:
    return ACTIVE_VECTOR_TABLE


def active_embedding_metadata(app_state=None) -> dict[str, int | str]:
    return {
        "embedding_model_id": getattr(app_state, "embedding_model_id", EMBEDDING_MODEL_ID) if app_state is not None else EMBEDDING_MODEL_ID,
        "embedding_dim": getattr(app_state, "embedding_dim", EMBEDDING_DIMENSION) if app_state is not None else EMBEDDING_DIMENSION,
    }


def create_conversation(conn: sqlite3.Connection, title: str | None = None) -> dict[str, Any]:
    import uuid

    now = int(time.time())
    conversation_id = str(uuid.uuid4())
    clean_title = (title or "New chat").strip()[:120] or "New chat"
    execute(
        conn,
        "INSERT INTO conversations (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
        (conversation_id, clean_title, now, now),
    )
    return {"id": conversation_id, "title": clean_title, "created_at": now, "updated_at": now}


def list_conversations(conn: sqlite3.Connection, limit: int = 50) -> list[dict[str, Any]]:
    rows = fetchall(
        conn,
        """
        SELECT id, title, created_at, updated_at
        FROM conversations
        WHERE archived = 0
        ORDER BY updated_at DESC
        LIMIT ?
        """,
        (limit,),
    )
    return [row_to_dict(row) for row in rows]


def append_message(
    conn: sqlite3.Connection,
    conversation_id: str,
    role: str,
    content: str,
    *,
    model: str | None = None,
    settings: dict[str, Any] | None = None,
    meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    import uuid

    now = int(time.time())
    message_id = str(uuid.uuid4())
    execute(
        conn,
        """
        INSERT INTO messages (id, conversation_id, role, content, model, settings_json, meta_json, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            message_id,
            conversation_id,
            role,
            content,
            model,
            json.dumps(settings or {}, separators=(",", ":")),
            json.dumps(meta or {}, separators=(",", ":")),
            now,
        ),
    )
    execute(conn, "UPDATE conversations SET updated_at = ? WHERE id = ?", (now, conversation_id))
    return {
        "id": message_id,
        "conversation_id": conversation_id,
        "role": role,
        "content": content,
        "model": model,
        "settings": settings or {},
        "meta": meta or {},
        "created_at": now,
    }


def save_message_sources(conn: sqlite3.Connection, message_id: str, sources: list[dict[str, Any]]) -> None:
    with SQLITE_LOCK:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM message_sources WHERE message_id = ?", (message_id,))
        for rank, source in enumerate(sources, start=1):
            cursor.execute(
                "INSERT INTO message_sources (message_id, source_rank, source_json) VALUES (?, ?, ?)",
                (message_id, rank, json.dumps(source, ensure_ascii=False, separators=(",", ":"))),
            )
        conn.commit()


def get_conversation(conn: sqlite3.Connection, conversation_id: str) -> dict[str, Any] | None:
    conversation = fetchone(conn, "SELECT id, title, created_at, updated_at FROM conversations WHERE id = ? AND archived = 0", (conversation_id,))
    if not conversation:
        return None
    message_rows = fetchall(
        conn,
        "SELECT * FROM messages WHERE conversation_id = ? ORDER BY created_at, rowid",
        (conversation_id,),
    )
    messages = []
    for row in message_rows:
        source_rows = fetchall(conn, "SELECT source_json FROM message_sources WHERE message_id = ? ORDER BY source_rank", (row["id"],))
        messages.append({
            "id": row["id"],
            "role": row["role"],
            "content": row["content"],
            "model": row["model"],
            "settings": json.loads(row["settings_json"] or "{}"),
            "meta": json.loads(row["meta_json"] or "{}"),
            "created_at": row["created_at"],
            "sources": [json.loads(source["source_json"]) for source in source_rows],
        })
    payload = row_to_dict(conversation)
    payload["messages"] = messages
    return payload


def rename_conversation(conn: sqlite3.Connection, conversation_id: str, title: str) -> dict[str, Any] | None:
    clean_title = title.strip()[:120]
    if not clean_title:
        return None
    execute(conn, "UPDATE conversations SET title = ?, updated_at = ? WHERE id = ?", (clean_title, int(time.time()), conversation_id))
    return get_conversation(conn, conversation_id)


def archive_conversation(conn: sqlite3.Connection, conversation_id: str) -> None:
    execute(conn, "UPDATE conversations SET archived = 1, updated_at = ? WHERE id = ?", (int(time.time()), conversation_id))
