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

    if not migration_applied(conn, "007_rag_observability"):
        for table, columns in {
            "documents": [
                ("text_hash", "TEXT"),
                ("parser_version", "TEXT"),
                ("chunking_profile", "TEXT"),
                ("chunking_config_hash", "TEXT"),
                ("embedding_config_hash", "TEXT"),
                ("parse_warnings", "TEXT"),
            ],
            "chunks": [
                ("block_type", "TEXT"),
                ("section_heading", "TEXT"),
                ("heading_path", "TEXT"),
                ("page_number", "INTEGER"),
                ("char_count", "INTEGER DEFAULT 0"),
                ("text_hash", "TEXT"),
                ("raw_text_hash", "TEXT"),
                ("contextual_text_hash", "TEXT"),
                ("chunking_profile", "TEXT"),
                ("chunking_config_hash", "TEXT"),
                ("parser_version", "TEXT"),
                ("embedded_at", "INTEGER"),
                ("embedding_status", "TEXT DEFAULT 'embedded'"),
                ("parse_warnings", "TEXT"),
            ],
        }.items():
            for column, definition in columns:
                add_column_if_missing(conn, table, column, definition)
        executescript(conn, """
            CREATE TABLE IF NOT EXISTS retrieval_queries (
                id TEXT PRIMARY KEY,
                raw_query TEXT NOT NULL,
                normalized_query TEXT NOT NULL,
                rewritten_query TEXT,
                retrieval_mode TEXT,
                created_at INTEGER NOT NULL,
                subqueries_json TEXT NOT NULL,
                no_answer_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS retrieval_candidates (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query_id TEXT NOT NULL REFERENCES retrieval_queries(id) ON DELETE CASCADE,
                stage TEXT NOT NULL,
                rank INTEGER NOT NULL,
                chunk_id TEXT,
                doc_id TEXT,
                source_filename TEXT,
                score REAL,
                vector_score REAL,
                bm25_score REAL,
                fusion_score REAL,
                rerank_score REAL,
                payload_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS retrieval_context (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query_id TEXT NOT NULL REFERENCES retrieval_queries(id) ON DELETE CASCADE,
                rank INTEGER NOT NULL,
                chunk_id TEXT,
                doc_id TEXT,
                source_id TEXT,
                context_text TEXT,
                payload_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS retrieval_latency (
                query_id TEXT PRIMARY KEY REFERENCES retrieval_queries(id) ON DELETE CASCADE,
                preprocessing_ms REAL DEFAULT 0,
                rewrite_ms REAL DEFAULT 0,
                vector_ms REAL DEFAULT 0,
                bm25_ms REAL DEFAULT 0,
                fusion_ms REAL DEFAULT 0,
                rerank_ms REAL DEFAULT 0,
                context_ms REAL DEFAULT 0,
                generation_ms REAL DEFAULT 0,
                total_ms REAL DEFAULT 0,
                payload_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS answer_records (
                id TEXT PRIMARY KEY,
                query_id TEXT REFERENCES retrieval_queries(id) ON DELETE SET NULL,
                conversation_id TEXT,
                message_id TEXT,
                answer_text TEXT,
                confidence REAL,
                support_status TEXT,
                created_at INTEGER NOT NULL,
                meta_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS answer_citations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                answer_id TEXT NOT NULL REFERENCES answer_records(id) ON DELETE CASCADE,
                chunk_id TEXT,
                source_id TEXT,
                support_status TEXT,
                payload_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS eval_runs (
                id TEXT PRIMARY KEY,
                pipeline TEXT NOT NULL,
                top_k INTEGER NOT NULL,
                created_at INTEGER NOT NULL,
                aggregate_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS eval_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL REFERENCES eval_runs(id) ON DELETE CASCADE,
                eval_id TEXT NOT NULL,
                question TEXT NOT NULL,
                metrics_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS user_feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query_id TEXT,
                message_id TEXT,
                feedback_value TEXT NOT NULL,
                failure_reason TEXT,
                correction_text TEXT,
                expected_doc_id TEXT,
                created_at INTEGER NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_retrieval_candidates_query ON retrieval_candidates(query_id, stage, rank);
            CREATE INDEX IF NOT EXISTS idx_retrieval_context_query ON retrieval_context(query_id, rank);
            CREATE INDEX IF NOT EXISTS idx_eval_results_run ON eval_results(run_id);
        """)
        mark_migration(conn, "007_rag_observability")

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


def save_retrieval_trace(conn: sqlite3.Connection, trace: dict[str, Any]) -> None:
    query_id = trace["query_id"]
    no_answer = trace.get("no_answer") or {}
    subqueries = trace.get("subqueries") or []
    with SQLITE_LOCK:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM retrieval_candidates WHERE query_id = ?", (query_id,))
        cursor.execute("DELETE FROM retrieval_context WHERE query_id = ?", (query_id,))
        cursor.execute("DELETE FROM retrieval_latency WHERE query_id = ?", (query_id,))
        cursor.execute("DELETE FROM retrieval_queries WHERE id = ?", (query_id,))
        cursor.execute(
            """
            INSERT INTO retrieval_queries (
                id, raw_query, normalized_query, rewritten_query, retrieval_mode,
                created_at, subqueries_json, no_answer_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                query_id,
                trace.get("raw_query", ""),
                trace.get("normalized_query", ""),
                trace.get("rewritten_query"),
                trace.get("retrieval_mode", ""),
                int(trace.get("timestamp") or time.time()),
                json.dumps(subqueries, ensure_ascii=False, separators=(",", ":")),
                json.dumps(no_answer, ensure_ascii=False, separators=(",", ":")),
            ),
        )
        stage_map = {
            "vector": trace.get("vector_candidates", []),
            "bm25": trace.get("bm25_candidates", []),
            "fused": trace.get("fused_candidates", []),
            "reranked": trace.get("reranked_candidates", []),
            "unused": trace.get("unused_candidates", []),
        }
        for stage, candidates in stage_map.items():
            for rank, candidate in enumerate(candidates, start=1):
                cursor.execute(
                    """
                    INSERT INTO retrieval_candidates (
                        query_id, stage, rank, chunk_id, doc_id, source_filename, score,
                        vector_score, bm25_score, fusion_score, rerank_score, payload_json
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        query_id,
                        stage,
                        int(candidate.get("rank") or rank),
                        candidate.get("chunk_id") or candidate.get("id"),
                        candidate.get("doc_id"),
                        candidate.get("doc_name") or candidate.get("source_filename"),
                        candidate.get("score"),
                        candidate.get("vector_score"),
                        candidate.get("lexical_score") if candidate.get("lexical_score") is not None else candidate.get("bm25_score"),
                        candidate.get("fusion_score"),
                        candidate.get("rerank_score"),
                        json.dumps(candidate, ensure_ascii=False, separators=(",", ":")),
                    ),
                )
        for rank, item in enumerate(trace.get("final_context", []), start=1):
            cursor.execute(
                """
                INSERT INTO retrieval_context (query_id, rank, chunk_id, doc_id, source_id, context_text, payload_json)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    query_id,
                    int(item.get("rank") or rank),
                    item.get("chunk_id"),
                    item.get("doc_id"),
                    item.get("source_id"),
                    item.get("text") or item.get("snippet"),
                    json.dumps(item, ensure_ascii=False, separators=(",", ":")),
                ),
            )
        latency = trace.get("latency") or {}
        cursor.execute(
            """
            INSERT INTO retrieval_latency (
                query_id, preprocessing_ms, rewrite_ms, vector_ms, bm25_ms, fusion_ms,
                rerank_ms, context_ms, generation_ms, total_ms, payload_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                query_id,
                latency.get("preprocessing_ms", 0),
                latency.get("rewrite_ms", 0),
                latency.get("vector_ms", 0),
                latency.get("bm25_ms", 0),
                latency.get("fusion_ms", 0),
                latency.get("rerank_ms", 0),
                latency.get("context_ms", 0),
                latency.get("generation_ms", 0),
                latency.get("total_ms", latency.get("retrieval_ms", 0)),
                json.dumps(latency, ensure_ascii=False, separators=(",", ":")),
            ),
        )
        conn.commit()


def list_retrieval_traces(conn: sqlite3.Connection, limit: int = 50) -> list[dict[str, Any]]:
    rows = fetchall(
        conn,
        """
        SELECT retrieval_queries.*, retrieval_latency.total_ms
        FROM retrieval_queries
        LEFT JOIN retrieval_latency ON retrieval_latency.query_id = retrieval_queries.id
        ORDER BY retrieval_queries.created_at DESC
        LIMIT ?
        """,
        (limit,),
    )
    return [
        {
            "query_id": row["id"],
            "raw_query": row["raw_query"],
            "normalized_query": row["normalized_query"],
            "retrieval_mode": row["retrieval_mode"],
            "created_at": row["created_at"],
            "total_ms": row["total_ms"],
            "no_answer": json.loads(row["no_answer_json"] or "{}"),
        }
        for row in rows
    ]


def get_retrieval_trace(conn: sqlite3.Connection, query_id: str) -> dict[str, Any] | None:
    row = fetchone(conn, "SELECT * FROM retrieval_queries WHERE id = ?", (query_id,))
    if not row:
        return None
    candidate_rows = fetchall(conn, "SELECT stage, payload_json FROM retrieval_candidates WHERE query_id = ? ORDER BY stage, rank", (query_id,))
    context_rows = fetchall(conn, "SELECT payload_json FROM retrieval_context WHERE query_id = ? ORDER BY rank", (query_id,))
    latency = fetchone(conn, "SELECT payload_json FROM retrieval_latency WHERE query_id = ?", (query_id,))
    candidates: dict[str, list[dict[str, Any]]] = {"vector": [], "bm25": [], "fused": [], "reranked": [], "unused": []}
    for candidate in candidate_rows:
        candidates.setdefault(candidate["stage"], []).append(json.loads(candidate["payload_json"]))
    return {
        "query_id": row["id"],
        "raw_query": row["raw_query"],
        "normalized_query": row["normalized_query"],
        "rewritten_query": row["rewritten_query"],
        "retrieval_mode": row["retrieval_mode"],
        "created_at": row["created_at"],
        "subqueries": json.loads(row["subqueries_json"] or "[]"),
        "no_answer": json.loads(row["no_answer_json"] or "{}"),
        "latency": json.loads(latency["payload_json"] or "{}") if latency else {},
        "candidates": candidates,
        "final_context": [json.loads(item["payload_json"]) for item in context_rows],
    }


def save_answer_record(conn: sqlite3.Connection, payload: dict[str, Any]) -> None:
    answer_id = payload["id"]
    with SQLITE_LOCK:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT OR REPLACE INTO answer_records (
                id, query_id, conversation_id, message_id, answer_text, confidence,
                support_status, created_at, meta_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                answer_id,
                payload.get("query_id"),
                payload.get("conversation_id"),
                payload.get("message_id"),
                payload.get("answer_text", ""),
                payload.get("confidence"),
                payload.get("support_status"),
                int(payload.get("created_at") or time.time()),
                json.dumps(payload.get("meta", {}), ensure_ascii=False, separators=(",", ":")),
            ),
        )
        cursor.execute("DELETE FROM answer_citations WHERE answer_id = ?", (answer_id,))
        for citation in payload.get("citations", []):
            cursor.execute(
                """
                INSERT INTO answer_citations (answer_id, chunk_id, source_id, support_status, payload_json)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    answer_id,
                    citation.get("chunk_id"),
                    citation.get("source_id"),
                    citation.get("status"),
                    json.dumps(citation, ensure_ascii=False, separators=(",", ":")),
                ),
            )
        conn.commit()


def save_eval_run(conn: sqlite3.Connection, payload: dict[str, Any]) -> None:
    with SQLITE_LOCK:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT OR REPLACE INTO eval_runs (id, pipeline, top_k, created_at, aggregate_json) VALUES (?, ?, ?, ?, ?)",
            (
                payload["id"],
                payload["pipeline"],
                payload["top_k"],
                payload["created_at"],
                json.dumps(payload.get("aggregate", {}), ensure_ascii=False, separators=(",", ":")),
            ),
        )
        cursor.execute("DELETE FROM eval_results WHERE run_id = ?", (payload["id"],))
        for result in payload.get("results", []):
            cursor.execute(
                "INSERT INTO eval_results (run_id, eval_id, question, metrics_json) VALUES (?, ?, ?, ?)",
                (
                    payload["id"],
                    result["eval_id"],
                    result.get("question", ""),
                    json.dumps(result.get("metrics", {}), ensure_ascii=False, separators=(",", ":")),
                ),
            )
        conn.commit()


def list_eval_runs(conn: sqlite3.Connection, limit: int = 50) -> list[dict[str, Any]]:
    rows = fetchall(conn, "SELECT * FROM eval_runs ORDER BY created_at DESC LIMIT ?", (limit,))
    return [
        {
            "id": row["id"],
            "pipeline": row["pipeline"],
            "top_k": row["top_k"],
            "created_at": row["created_at"],
            "aggregate": json.loads(row["aggregate_json"] or "{}"),
        }
        for row in rows
    ]


def get_eval_run(conn: sqlite3.Connection, run_id: str) -> dict[str, Any] | None:
    row = fetchone(conn, "SELECT * FROM eval_runs WHERE id = ?", (run_id,))
    if not row:
        return None
    result_rows = fetchall(conn, "SELECT * FROM eval_results WHERE run_id = ? ORDER BY id", (run_id,))
    return {
        "id": row["id"],
        "pipeline": row["pipeline"],
        "top_k": row["top_k"],
        "created_at": row["created_at"],
        "aggregate": json.loads(row["aggregate_json"] or "{}"),
        "results": [
            {
                "eval_id": item["eval_id"],
                "question": item["question"],
                "metrics": json.loads(item["metrics_json"] or "{}"),
            }
            for item in result_rows
        ],
    }
