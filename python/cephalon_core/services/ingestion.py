import asyncio
from dataclasses import dataclass, field
import json
import os
import re
import time
import uuid
from collections.abc import Awaitable, Callable

from .. import storage
from ..schemas import RagSettings
from . import documents
from . import observability
from .retrieval import ensure_vector_table, get_embedding, get_embeddings, vector_table_name
from .pdf_parser import DocumentBlock

PARENT_TARGET_TOKENS = 520
PARENT_MAX_TOKENS = 650
CHILD_TARGET_TOKENS = 110
CHILD_MAX_TOKENS = 150
PARSER_VERSION = "cephalon-basic-2026-05"
CHUNKING_PROFILE = "semantic_parent_child_v1"


ProgressCallback = Callable[[str, int], Awaitable[None]]


@dataclass
class ParentDraft:
    text: str
    blocks: list[DocumentBlock] = field(default_factory=list)


@dataclass
class ChildDraft:
    text: str
    block_type: str = "paragraph"
    heading_path: list[str] = field(default_factory=list)
    page_number: int | None = None
    page_end: int | None = None
    block_index: int | None = None
    bounding_box: tuple[float, float, float, float] | None = None
    provenance: dict = field(default_factory=dict)

    @property
    def section_heading(self) -> str | None:
        return self.heading_path[-1] if self.heading_path else None


async def process_single_file(
    app_state,
    file_path: str,
    rag_settings: RagSettings,
    *,
    force_text: bool = False,
    existing_doc_id: str | None = None,
    progress: ProgressCallback | None = None,
) -> dict:
    if not os.path.isfile(file_path):
        return {"status": "failed", "path": file_path, "error": "Path is not a file."}
    if not documents.collect_supported_files(file_path, force_text=force_text):
        return {"status": "failed", "path": file_path, "error": "Unsupported file type."}

    doc_id = existing_doc_id or str(uuid.uuid4())
    persistence_started = False
    try:
        content_hash = await asyncio.to_thread(documents.get_file_hash, file_path)
        existing = documents.find_existing_doc_by_hash(app_state.sqlite, content_hash)
        if existing and existing_doc_id != existing["id"]:
            return {"status": "skipped", "path": file_path, "doc_id": existing["id"], "reason": "duplicate"}

        await _report_progress(progress, "extracting", 10)
        extracted = await asyncio.to_thread(documents.extract_document, file_path, force_text=force_text)
        raw_text = extracted.text
        extraction_mode = extracted.extraction_mode
        parser_version = extracted.parser_version
        parse_warnings = extracted.warnings
        metadata = storage.active_embedding_metadata(app_state)
        chunking_config = _chunking_config(rag_settings)
        chunking_hash = observability.chunking_config_hash(CHUNKING_PROFILE, chunking_config)
        text_hash = observability.text_hash(raw_text)
        if not raw_text.strip():
            raise ValueError("No extractable text found.")

        await _report_progress(progress, "chunking", 35)
        if extraction_mode == "native_structured":
            parents = build_structured_parent_chunks(extracted.blocks, rag_settings)
        else:
            parents = [ParentDraft(text=text) for text in build_parent_chunks(raw_text, rag_settings)]
        if not parents:
            raise ValueError("No text chunks produced.")

        lance_data = []
        vector_texts = []
        parent_rows = []
        summary_rows = []
        chunk_rows = []
        fts_rows = []
        child_count = 0
        now = int(time.time())
        for parent_index, parent in enumerate(parents):
            parent_text = parent.text
            parent_id = f"{doc_id}_p{parent_index}"
            summary = summarize_parent(parent_text)
            summary_id = f"{parent_id}_s"
            parent_rows.append((parent_id, doc_id, parent_index, parent_text, summary, estimate_tokens(parent_text), now))
            summary_rows.append((summary_id, doc_id, parent_id, parent_index, summary, estimate_tokens(summary), now))

            vector_texts.append(summary)
            lance_data.append({
                "id": summary_id,
                "doc_id": doc_id,
                "text": summary,
                "chunk_index": -100000 - parent_index,
                "parent_id": parent_id,
                "source_kind": "summary",
                **metadata,
                "chunk_length": len(summary),
            })

            if parent.blocks:
                child_chunks = await build_structured_child_chunks(app_state, parent.blocks, rag_settings)
            else:
                child_chunks = [
                    ChildDraft(text=text)
                    for text in await build_semantic_child_chunks(app_state, parent_text, rag_settings)
                ]
            for child in child_chunks:
                child_text = child.text
                chunk_id = f"{doc_id}_{child_count}"
                token_count = estimate_tokens(child_text)
                child_hash = observability.text_hash(child_text)
                contextual_text = contextualize_chunk(
                    child_text,
                    os.path.basename(file_path),
                    " > ".join(child.heading_path) or None,
                    child.block_type,
                    page_number=child.page_number,
                    page_end=child.page_end,
                )
                chunk_rows.append((
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
                    child.block_type,
                    child.section_heading,
                    json.dumps(child.heading_path, ensure_ascii=False) if child.heading_path else None,
                    child.page_number,
                    child.page_end,
                    child.block_index,
                    json.dumps(child.bounding_box) if child.bounding_box else None,
                    json.dumps(child.provenance, ensure_ascii=False) if child.provenance else None,
                    len(child_text),
                    child_hash,
                    child_hash,
                    observability.text_hash(contextual_text),
                    CHUNKING_PROFILE,
                    chunking_hash,
                    parser_version,
                    now,
                    "embedded",
                    json.dumps(parse_warnings, ensure_ascii=False) if parse_warnings else None,
                ))
                fts_rows.append((chunk_id, doc_id, child_text))
                # Preserve raw text for FTS, source display, and parent context,
                # but give dense retrieval concise document context.
                vector_texts.append(contextual_text)
                lance_data.append({
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

        await _report_progress(progress, "embedding", 65)
        vectors = await _embed_many(app_state, vector_texts)
        for row, vector in zip(lance_data, vectors, strict=True):
            row["vector"] = vector

        await _report_progress(progress, "persisting", 90)
        previous_vectors = None
        if existing_doc_id:
            previous_vectors = await asyncio.to_thread(_replace_document_vectors, app_state, doc_id, lance_data)
        else:
            await asyncio.to_thread(ensure_vector_table, app_state, lance_data)

        persistence_started = True
        try:
            documents.register_ingesting_document(app_state.sqlite, file_path, content_hash, extraction_mode, doc_id, metadata)
            _replace_document_rows(
                app_state.sqlite,
                doc_id,
                parent_rows,
                summary_rows,
                chunk_rows,
                fts_rows,
                text_hash=text_hash,
                chunking_hash=chunking_hash,
                embedding_config_hash=f"{metadata['embedding_model_id']}:{metadata['embedding_dim']}",
                parser_version=parser_version,
                parse_warnings=parse_warnings,
            )
        except Exception:
            if existing_doc_id:
                await asyncio.to_thread(_restore_document_vectors, app_state, doc_id, previous_vectors or [])
            else:
                await asyncio.to_thread(delete_document_vectors, app_state, doc_id)
            raise

        documents.mark_document_ready(app_state.sqlite, doc_id, child_count)
        await _report_progress(progress, "complete", 100)
        return {"status": "ready", "path": file_path, "doc_id": doc_id, "chunks": child_count, "extraction_mode": extraction_mode}
    except Exception as exc:
        if existing_doc_id:
            previous_count = storage.fetchone(app_state.sqlite, "SELECT COUNT(*) AS count FROM chunks WHERE doc_id = ?", (doc_id,))
            storage.execute(
                app_state.sqlite,
                "UPDATE documents SET status = 'ready', chunk_count = ?, last_error = ? WHERE id = ?",
                (previous_count["count"] if previous_count else 0, str(exc), doc_id),
            )
        elif persistence_started:
            documents.mark_document_failed(app_state.sqlite, doc_id, str(exc))
        return {"status": "failed", "path": file_path, "doc_id": doc_id, "error": str(exc)}


def _replace_document_rows(
    sqlite_conn,
    doc_id: str,
    parent_rows: list[tuple],
    summary_rows: list[tuple],
    chunk_rows: list[tuple],
    fts_rows: list[tuple],
    *,
    text_hash: str,
    chunking_hash: str,
    embedding_config_hash: str,
    parser_version: str,
    parse_warnings: list[str],
) -> None:
    with storage.SQLITE_LOCK:
        cursor = sqlite_conn.cursor()
        try:
            cursor.execute("BEGIN")
            cursor.execute("DELETE FROM chunks_fts WHERE doc_id = ?", (doc_id,))
            cursor.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
            cursor.execute("DELETE FROM summary_nodes WHERE doc_id = ?", (doc_id,))
            cursor.execute("DELETE FROM parent_chunks WHERE doc_id = ?", (doc_id,))
            cursor.executemany(
                """
                INSERT INTO parent_chunks (id, doc_id, parent_index, text, summary, token_count, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                parent_rows,
            )
            cursor.executemany(
                """
                INSERT INTO summary_nodes (id, doc_id, parent_id, chunk_index, summary, token_count, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                summary_rows,
            )
            cursor.executemany(
                """
                INSERT INTO chunks (
                    id, doc_id, chunk_index, text, parent_id, summary_id, token_count,
                    semantic_role, chunk_length, embedding_model_id, embedding_dim,
                    block_type, section_heading, heading_path, page_number, page_end,
                    block_index, bounding_box, provenance_json, char_count, text_hash,
                    raw_text_hash, contextual_text_hash, chunking_profile,
                    chunking_config_hash, parser_version, embedded_at, embedding_status,
                    parse_warnings
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                chunk_rows,
            )
            cursor.executemany(
                "INSERT INTO chunks_fts (chunk_id, doc_id, text) VALUES (?, ?, ?)",
                fts_rows,
            )
            cursor.execute(
                """
                UPDATE documents
                SET text_hash = ?, parser_version = ?, chunking_profile = ?,
                    chunking_config_hash = ?, embedding_config_hash = ?, parse_warnings = ?
                WHERE id = ?
                """,
                (
                    text_hash,
                    parser_version,
                    CHUNKING_PROFILE,
                    chunking_hash,
                    embedding_config_hash,
                    json.dumps(parse_warnings, ensure_ascii=False) if parse_warnings else None,
                    doc_id,
                ),
            )
            sqlite_conn.commit()
        except Exception:
            sqlite_conn.rollback()
            raise


async def _embed_many(app_state, texts: list[str]) -> list[list[float]]:
    if getattr(app_state, "embedder", None) is None:
        return [await get_embedding(app_state, text) for text in texts]
    return await get_embeddings(app_state, texts)


async def _report_progress(progress: ProgressCallback | None, stage: str, percent: int) -> None:
    if progress is not None:
        await progress(stage, percent)


def _replace_document_vectors(app_state, doc_id: str, rows: list[dict]) -> list[dict]:
    table_name = vector_table_name(app_state)
    if table_name not in app_state.lance.table_names():
        ensure_vector_table(app_state, rows)
        return []

    table = app_state.lance.open_table(table_name)
    previous_rows = _document_vector_rows(table, doc_id)
    table.delete(f"doc_id = {quote_lance_string(doc_id)}")
    try:
        table.add(rows)
    except Exception:
        if previous_rows:
            table.add(previous_rows)
        raise
    return previous_rows


def _restore_document_vectors(app_state, doc_id: str, rows: list[dict]) -> None:
    delete_document_vectors(app_state, doc_id)
    if rows:
        ensure_vector_table(app_state, rows)


def _document_vector_rows(table, doc_id: str) -> list[dict]:
    if hasattr(table, "rows"):
        return [dict(row) for row in table.rows if row.get("doc_id") == doc_id]
    return (
        table.search()
        .where(f"doc_id = {quote_lance_string(doc_id)}")
        .limit(100_000)
        .to_list()
    )


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


def _chunking_config(settings: RagSettings) -> dict[str, int]:
    return {
        "parent_target_tokens": settings.parent_target_tokens,
        "parent_max_tokens": settings.parent_max_tokens,
        "child_target_tokens": settings.child_target_tokens,
        "child_max_tokens": settings.child_max_tokens,
        "child_overlap_tokens": settings.child_overlap_tokens,
    }


def parser_version_for_path(path: str) -> str:
    return documents.PDF_PARSER_VERSION if os.path.splitext(path)[1].lower() == ".pdf" else PARSER_VERSION


def refresh_document_staleness(app_state, rag_settings: RagSettings | None = None) -> dict:
    """Refresh persisted stale-index flags without reparsing or re-embedding files."""
    rag_settings = rag_settings or storage.get_rag_settings(app_state.sqlite)
    embedding_metadata_known = (
        hasattr(app_state, "embedding_model_id")
        and hasattr(app_state, "embedding_dim")
    )
    metadata = storage.active_embedding_metadata(app_state)
    chunking_hash = observability.chunking_config_hash(CHUNKING_PROFILE, _chunking_config(rag_settings))
    embedding_config_hash = f"{metadata['embedding_model_id']}:{metadata['embedding_dim']}"
    rows = storage.fetchall(
        app_state.sqlite,
        "SELECT * FROM documents WHERE type = 'file' AND status = 'ready'",
    )
    stale_count = 0
    reasons_by_document: dict[str, list[str]] = {}

    for row in rows:
        stored = storage.row_to_dict(row) or {}
        path = str(stored.get("path") or "")
        reasons: list[str] = []
        current_size = stored.get("size_bytes")
        current_modified = stored.get("modified_at")

        if not os.path.isfile(path):
            reasons.append("file_missing")
            current_content_hash = stored.get("content_hash")
        else:
            try:
                current_size, current_modified = documents.file_metadata(path)
                metadata_changed = (
                    current_size != stored.get("size_bytes")
                    or current_modified != stored.get("modified_at")
                )
                current_content_hash = (
                    documents.get_file_hash(path)
                    if metadata_changed
                    else stored.get("content_hash")
                )
            except OSError:
                current_content_hash = stored.get("content_hash")
                reasons.append("file_unreadable")

        current = {
            "content_hash": current_content_hash,
            "chunking_config_hash": chunking_hash,
            "parser_version": parser_version_for_path(path),
            "embedding_model_id": (
                metadata["embedding_model_id"]
                if embedding_metadata_known
                else stored.get("embedding_model_id")
            ),
            "embedding_config_hash": (
                embedding_config_hash
                if embedding_metadata_known
                else stored.get("embedding_config_hash")
            ),
        }
        detected = observability.detect_stale_state(stored, current)
        reasons.extend(reason for reason in detected["reasons"] if reason not in reasons)
        is_stale = bool(reasons)
        if is_stale:
            stale_count += 1
            reasons_by_document[row["id"]] = reasons
        storage.execute(
            app_state.sqlite,
            """
            UPDATE documents
            SET stale_embedding = ?, stale_reasons = ?, size_bytes = ?, modified_at = ?
            WHERE id = ?
            """,
            (
                int(is_stale),
                json.dumps(reasons, separators=(",", ":")) if reasons else None,
                current_size,
                current_modified,
                row["id"],
            ),
        )

    return {
        "checked_document_count": len(rows),
        "stale_document_count": stale_count,
        "reasons_by_document": reasons_by_document,
    }


def build_structured_parent_chunks(
    blocks: list[DocumentBlock],
    settings: RagSettings | None = None,
) -> list[ParentDraft]:
    target = settings.parent_target_tokens if settings else PARENT_TARGET_TOKENS
    maximum = settings.parent_max_tokens if settings else PARENT_MAX_TOKENS
    target = min(target, maximum)
    eligible = [
        block
        for block in blocks
        if block.text.strip() and block.block_type not in {"header", "footer"}
    ]
    parents: list[ParentDraft] = []
    current: list[DocumentBlock] = []
    current_tokens = 0
    current_path: tuple[str, ...] = ()

    def flush() -> None:
        nonlocal current, current_tokens, current_path
        if current:
            parents.append(ParentDraft(
                text="\n\n".join(block.text.strip() for block in current),
                blocks=list(current),
            ))
        current = []
        current_tokens = 0
        current_path = ()

    for block in eligible:
        block_tokens = estimate_tokens(block.text)
        block_path = tuple(block.heading_path)
        section_changed = (
            current
            and block_path
            and current_path
            and block_path != current_path
            and current_tokens >= max(64, target // 3)
        )
        if current and (current_tokens + block_tokens > maximum or section_changed):
            flush()
        current.append(block)
        current_tokens += block_tokens
        if block_path:
            current_path = block_path
        if current_tokens >= target and block.block_type not in {"title", "heading"}:
            flush()
    flush()
    return parents


async def build_structured_child_chunks(
    app_state,
    blocks: list[DocumentBlock],
    settings: RagSettings | None = None,
) -> list[ChildDraft]:
    target = settings.child_target_tokens if settings else CHILD_TARGET_TOKENS
    maximum = settings.child_max_tokens if settings else CHILD_MAX_TOKENS
    target = min(target, maximum)
    drafts: list[ChildDraft] = []
    current: list[DocumentBlock] = []
    current_tokens = 0

    def flush() -> None:
        nonlocal current, current_tokens
        if current:
            drafts.append(_child_from_blocks(current))
        current = []
        current_tokens = 0

    for block in blocks:
        if block.block_type in {"title", "heading", "header", "footer"}:
            continue
        if block.block_type == "table":
            flush()
            drafts.extend(_table_child_drafts(block, maximum))
            continue
        block_tokens = estimate_tokens(block.text)
        if block_tokens > maximum:
            flush()
            pieces = await build_semantic_child_chunks(app_state, block.text, settings)
            drafts.extend(_child_from_blocks([block], text=piece) for piece in pieces)
            continue
        incompatible = (
            current
            and (
                current[-1].heading_path != block.heading_path
                or current[-1].page_number != block.page_number
                or current[-1].block_type != block.block_type
                or current_tokens + block_tokens > maximum
            )
        )
        if incompatible:
            flush()
        current.append(block)
        current_tokens += block_tokens
        if current_tokens >= target or block.block_type in {"caption", "footnote"}:
            flush()
    flush()
    return drafts


def _child_from_blocks(blocks: list[DocumentBlock], *, text: str | None = None) -> ChildDraft:
    page_numbers = [block.page_number for block in blocks]
    block_types = {block.block_type for block in blocks}
    paths = [block.heading_path for block in blocks if block.heading_path]
    same_page = len(set(page_numbers)) == 1
    boxes = [block.bounding_box for block in blocks if block.bounding_box]
    bounding_box = None
    if same_page and boxes:
        bounding_box = (
            min(box[0] for box in boxes),
            min(box[1] for box in boxes),
            max(box[2] for box in boxes),
            max(box[3] for box in boxes),
        )
    return ChildDraft(
        text=(text if text is not None else "\n".join(block.text.strip() for block in blocks)).strip(),
        block_type=next(iter(block_types)) if len(block_types) == 1 else "mixed",
        heading_path=list(paths[0]) if paths else [],
        page_number=min(page_numbers) if page_numbers else None,
        page_end=max(page_numbers) if page_numbers else None,
        block_index=min((block.block_index for block in blocks), default=None),
        bounding_box=bounding_box,
        provenance={
            "source_block_indices": [block.block_index for block in blocks],
            "source_block_types": [block.block_type for block in blocks],
        },
    )


def _table_child_drafts(block: DocumentBlock, maximum: int) -> list[ChildDraft]:
    lines = [line.strip() for line in block.text.splitlines() if line.strip()]
    if not lines or estimate_tokens(block.text) <= maximum:
        return [_child_from_blocks([block])]
    header = lines[0]
    drafts: list[ChildDraft] = []
    rows: list[str] = []
    tokens = estimate_tokens(header)
    for row in lines[1:]:
        row_tokens = estimate_tokens(row)
        if rows and tokens + row_tokens > maximum:
            drafts.append(_child_from_blocks([block], text="\n".join([header, *rows])))
            rows = []
            tokens = estimate_tokens(header)
        rows.append(row)
        tokens += row_tokens
    if rows:
        drafts.append(_child_from_blocks([block], text="\n".join([header, *rows])))
    for index, draft in enumerate(drafts):
        draft.provenance["table_part"] = index + 1
        draft.provenance["table_parts"] = len(drafts)
    return drafts


def build_parent_chunks(text: str, settings: RagSettings | None = None) -> list[str]:
    target = settings.parent_target_tokens if settings else PARENT_TARGET_TOKENS
    maximum = settings.parent_max_tokens if settings else PARENT_MAX_TOKENS
    if target > maximum:
        target = maximum
    units = split_text_units(text)
    parents: list[str] = []
    current: list[str] = []
    current_tokens = 0
    for unit in units:
        unit_tokens = estimate_tokens(unit)
        if current and current_tokens + unit_tokens > maximum:
            parents.append("\n".join(current).strip())
            current = []
            current_tokens = 0
        current.append(unit)
        current_tokens += unit_tokens
        if current_tokens >= target:
            parents.append("\n".join(current).strip())
            current = []
            current_tokens = 0
    if current:
        parents.append("\n".join(current).strip())
    return parents


async def build_semantic_child_chunks(app_state, parent_text: str, settings: RagSettings | None = None) -> list[str]:
    target = settings.child_target_tokens if settings else CHILD_TARGET_TOKENS
    maximum = settings.child_max_tokens if settings else CHILD_MAX_TOKENS
    overlap = settings.child_overlap_tokens if settings else 0
    if target > maximum:
        target = maximum
    units = split_text_units(parent_text)
    if not units:
        return []
    # Keep child chunks aligned to sentence/paragraph units, but do not call
    # the embedder while deciding their boundaries.  That used to add many
    # duplicate ONNX passes before the final batched indexing pass and made
    # large real-world documents impractically slow to ingest.
    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0
    for unit in units:
        unit_tokens = estimate_tokens(unit)
        should_break = current and current_tokens + unit_tokens > maximum
        if should_break:
            chunks.append(" ".join(current).strip())
            overlap_words = " ".join(current).split()[-overlap:] if overlap else []
            current = [" ".join(overlap_words)] if overlap_words else []
            current_tokens = len(overlap_words)
        current.append(unit)
        current_tokens += unit_tokens
        if current_tokens >= target:
            chunks.append(" ".join(current).strip())
            overlap_words = " ".join(current).split()[-overlap:] if overlap else []
            current = [" ".join(overlap_words)] if overlap_words else []
            current_tokens = len(overlap_words)
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


def contextualize_chunk(
    chunk_text: str,
    title: str,
    heading_path: str | None,
    block_type: str,
    *,
    page_number: int | None = None,
    page_end: int | None = None,
) -> str:
    parts = [f"Document: {title}"]
    if heading_path:
        parts.append(f"Section: {heading_path}")
    if page_number is not None:
        page_label = str(page_number)
        if page_end is not None and page_end != page_number:
            page_label = f"{page_label}-{page_end}"
        parts.append(f"Page: {page_label}")
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
