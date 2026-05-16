import os
import re
import time

from .. import storage
from ..schemas import RagSettings
from . import documents
from . import observability
from .retrieval import ensure_vector_table, get_embedding, vector_table_name

PARENT_TARGET_TOKENS = 520
PARENT_MAX_TOKENS = 650
CHILD_TARGET_TOKENS = 110
CHILD_MAX_TOKENS = 150
PARSER_VERSION = "cephalon-basic-2026-05"
CHUNKING_PROFILE = "semantic_parent_child_v1"


async def process_single_file(app_state, file_path: str, rag_settings: RagSettings, *, force_text: bool = False, existing_doc_id: str | None = None) -> dict:
    if not os.path.isfile(file_path):
        return {"status": "failed", "path": file_path, "error": "Path is not a file."}
    if not documents.collect_supported_files(file_path, force_text=force_text):
        return {"status": "failed", "path": file_path, "error": "Unsupported file type."}

    doc_id = None
    try:
        content_hash = documents.get_file_hash(file_path)
        existing = documents.find_existing_doc_by_hash(app_state.sqlite, content_hash)
        if existing and existing_doc_id != existing["id"]:
            return {"status": "skipped", "path": file_path, "doc_id": existing["id"], "reason": "duplicate"}

        raw_text, extraction_mode = documents.extract_text(file_path, force_text=force_text)
        metadata = storage.active_embedding_metadata(app_state)
        doc_id = documents.register_ingesting_document(app_state.sqlite, file_path, content_hash, extraction_mode, existing_doc_id, metadata)
        chunking_hash = observability.chunking_config_hash(CHUNKING_PROFILE, {
            "parent_target_tokens": PARENT_TARGET_TOKENS,
            "parent_max_tokens": PARENT_MAX_TOKENS,
            "child_target_tokens": CHILD_TARGET_TOKENS,
            "child_max_tokens": CHILD_MAX_TOKENS,
        })
        text_hash = observability.text_hash(raw_text)
        storage.execute(
            app_state.sqlite,
            """
            UPDATE documents
            SET text_hash = ?, parser_version = ?, chunking_profile = ?,
                chunking_config_hash = ?, embedding_config_hash = ?, parse_warnings = NULL
            WHERE id = ?
            """,
            (
                text_hash,
                PARSER_VERSION,
                CHUNKING_PROFILE,
                chunking_hash,
                f"{metadata['embedding_model_id']}:{metadata['embedding_dim']}",
                doc_id,
            ),
        )
        storage.delete_document_fts(app_state.sqlite, doc_id)
        storage.delete_document_hierarchy(app_state.sqlite, doc_id)
        storage.execute(app_state.sqlite, "DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
        if not raw_text.strip():
            raise ValueError("No extractable text found.")

        parents = build_parent_chunks(raw_text)
        if not parents:
            raise ValueError("No text chunks produced.")

        lance_data = []
        child_count = 0
        now = int(time.time())
        for parent_index, parent_text in enumerate(parents):
            parent_id = f"{doc_id}_p{parent_index}"
            summary = summarize_parent(parent_text)
            summary_id = f"{parent_id}_s"
            storage.execute(
                app_state.sqlite,
                """
                INSERT INTO parent_chunks (id, doc_id, parent_index, text, summary, token_count, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (parent_id, doc_id, parent_index, parent_text, summary, estimate_tokens(parent_text), now),
            )
            storage.execute(
                app_state.sqlite,
                """
                INSERT INTO summary_nodes (id, doc_id, parent_id, chunk_index, summary, token_count, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (summary_id, doc_id, parent_id, parent_index, summary, estimate_tokens(summary), now),
            )

            summary_vector = await get_embedding(app_state, summary)
            lance_data.append({
                "vector": summary_vector,
                "id": summary_id,
                "doc_id": doc_id,
                "text": summary,
                "chunk_index": -100000 - parent_index,
                "parent_id": parent_id,
                "source_kind": "summary",
                **metadata,
                "chunk_length": len(summary),
            })

            child_chunks = await build_semantic_child_chunks(app_state, parent_text)
            for child_text in child_chunks:
                chunk_id = f"{doc_id}_{child_count}"
                vector = await get_embedding(app_state, child_text)
                token_count = estimate_tokens(child_text)
                child_hash = observability.text_hash(child_text)
                contextual_text = contextualize_chunk(child_text, os.path.basename(file_path), None, "paragraph")
                storage.execute(
                    app_state.sqlite,
                    """
                    INSERT INTO chunks (
                        id, doc_id, chunk_index, text, parent_id, summary_id, token_count,
                        semantic_role, chunk_length, embedding_model_id, embedding_dim,
                        block_type, char_count, text_hash, raw_text_hash, contextual_text_hash,
                        chunking_profile, chunking_config_hash, parser_version, embedded_at, embedding_status
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        chunk_id,
                        doc_id,
                        child_count,
                        child_text,
                        parent_id,
                        summary_id,
                        token_count,
                        "child",
                        len(child_text),
                        metadata["embedding_model_id"],
                        metadata["embedding_dim"],
                        "paragraph",
                        len(child_text),
                        child_hash,
                        child_hash,
                        observability.text_hash(contextual_text),
                        CHUNKING_PROFILE,
                        chunking_hash,
                        PARSER_VERSION,
                        now,
                        "embedded",
                    ),
                )
                storage.upsert_chunk_fts(app_state.sqlite, chunk_id, doc_id, child_text)
                lance_data.append({
                    "vector": vector,
                    "id": chunk_id,
                    "doc_id": doc_id,
                    "text": child_text,
                    "chunk_index": child_count,
                    "parent_id": parent_id,
                    "source_kind": "child",
                    **metadata,
                    "chunk_length": len(child_text),
                })
                child_count += 1

        if child_count == 0:
            raise ValueError("No text chunks produced.")

        ensure_vector_table(app_state, lance_data)

        documents.mark_document_ready(app_state.sqlite, doc_id, child_count)
        return {"status": "ready", "path": file_path, "doc_id": doc_id, "chunks": child_count, "extraction_mode": extraction_mode}
    except Exception as exc:
        if doc_id:
            documents.mark_document_failed(app_state.sqlite, doc_id, str(exc))
        return {"status": "failed", "path": file_path, "doc_id": doc_id, "error": str(exc)}


def estimate_tokens(text: str) -> int:
    return max(1, len(re.findall(r"\S+", text)))


def split_text_units(text: str) -> list[str]:
    units: list[str] = []
    for block in re.split(r"\n\s*\n", text.strip()):
        clean = block.strip()
        if not clean:
            continue
        lines = [line.strip() for line in clean.splitlines() if line.strip()]
        if len(lines) > 1 and _looks_like_row_block(lines):
            units.extend(lines)
            continue
        units.extend(sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+|\n+", clean) if sentence.strip())
    return units


def _looks_like_row_block(lines: list[str]) -> bool:
    row_like = 0
    for line in lines:
        if re.search(r"\d", line) and ("," in line or "\t" in line or "/" in line or re.search(r"\s+\d", line)):
            row_like += 1
    return row_like >= max(2, len(lines) // 2)


def build_parent_chunks(text: str) -> list[str]:
    units = split_text_units(text)
    parents: list[str] = []
    current: list[str] = []
    current_tokens = 0
    for unit in units:
        unit_tokens = estimate_tokens(unit)
        if current and current_tokens + unit_tokens > PARENT_MAX_TOKENS:
            parents.append("\n".join(current).strip())
            current = []
            current_tokens = 0
        current.append(unit)
        current_tokens += unit_tokens
        if current_tokens >= PARENT_TARGET_TOKENS:
            parents.append("\n".join(current).strip())
            current = []
            current_tokens = 0
    if current:
        parents.append("\n".join(current).strip())
    return parents


async def build_semantic_child_chunks(app_state, parent_text: str) -> list[str]:
    units = split_text_units(parent_text)
    if not units:
        return []
    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0
    current_vector: list[float] | None = None
    for unit in units:
        unit_tokens = estimate_tokens(unit)
        should_break = current and current_tokens + unit_tokens > CHILD_MAX_TOKENS
        if current and current_tokens >= 60 and not should_break:
            next_vector = await get_embedding(app_state, unit)
            if current_vector is None:
                current_vector = await get_embedding(app_state, " ".join(current))
            should_break = cosine_similarity(current_vector, next_vector) < 0.18
        if should_break:
            chunks.append(" ".join(current).strip())
            current = []
            current_tokens = 0
            current_vector = None
        current.append(unit)
        current_tokens += unit_tokens
        if current_tokens >= CHILD_TARGET_TOKENS:
            chunks.append(" ".join(current).strip())
            current = []
            current_tokens = 0
            current_vector = None
    if current:
        chunks.append(" ".join(current).strip())
    return chunks


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = sum(a * a for a in left) ** 0.5
    right_norm = sum(b * b for b in right) ** 0.5
    if not left_norm or not right_norm:
        return 0.0
    return dot / (left_norm * right_norm)


def summarize_parent(text: str) -> str:
    units = split_text_units(text)
    if not units:
        return text[:500]
    selected = units[:3]
    summary = " ".join(selected)
    return summary[:700]


def contextualize_chunk(chunk_text: str, title: str, heading_path: str | None, block_type: str) -> str:
    parts = [f"Document: {title}"]
    if heading_path:
        parts.append(f"Section: {heading_path}")
    parts.append(f"Block type: {block_type}")
    return "\n".join(parts) + "\n\n" + chunk_text.strip()


async def process_directory(app_state, dir_path: str, rag_settings: RagSettings, *, force_text: bool = False) -> list[dict]:
    results = []
    for file_path in documents.collect_supported_files(dir_path, force_text=force_text):
        results.append(await process_single_file(app_state, file_path, rag_settings, force_text=force_text))
    return results


def delete_document_vectors(app_state, doc_id: str) -> None:
    table_name = vector_table_name(app_state)
    if table_name in app_state.lance.table_names():
        app_state.lance.open_table(table_name).delete(f"doc_id = {quote_lance_string(doc_id)}")


def quote_lance_string(value: str) -> str:
    return "'" + value.replace("\\", "\\\\").replace("'", "\\'") + "'"


def delete_document_rows(app_state, doc_id: str) -> None:
    with storage.SQLITE_LOCK:
        cursor = app_state.sqlite.cursor()
        cursor.execute("DELETE FROM chunks_fts WHERE doc_id = ?", (doc_id,))
        cursor.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
        cursor.execute("DELETE FROM summary_nodes WHERE doc_id = ?", (doc_id,))
        cursor.execute("DELETE FROM parent_chunks WHERE doc_id = ?", (doc_id,))
        cursor.execute("DELETE FROM document_tags WHERE doc_id = ?", (doc_id,))
        cursor.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
        app_state.sqlite.commit()


def mark_reindexing(app_state, doc_id: str) -> str:
    row = storage.fetchone(app_state.sqlite, "SELECT path FROM documents WHERE id = ? AND type = 'file'", (doc_id,))
    if not row:
        raise ValueError("Document not found.")
    storage.execute(app_state.sqlite, "UPDATE documents SET status = 'queued', last_error = NULL WHERE id = ?", (doc_id,))
    return row["path"]
