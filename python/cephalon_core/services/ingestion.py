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
from . import document_assets
from . import observability
from .retrieval import ensure_vector_table, get_embedding, get_embeddings, vector_table_name
from .pdf_parser import DocumentBlock

PARENT_TARGET_TOKENS = 520
PARENT_MAX_TOKENS = 650
CHILD_TARGET_TOKENS = 110
CHILD_MAX_TOKENS = 150
PARSER_VERSION = "cephalon-basic-2026-05"
CHUNKING_PROFILE = "semantic_parent_child_v2"
SUMMARY_PROFILE = "deterministic_parent_summary_v2"
SUMMARY_MAX_CHARACTERS = 700
SUMMARY_MAX_CONTENT_UNITS = 5
SUMMARY_MAX_ENTITIES = 6
SUMMARY_MAX_METADATA_CHARACTERS = 110
SUMMARY_MIN_UNIT_CHARACTERS = 56
SUMMARY_MAX_UNIT_CHARACTERS = 180

# HiChunk and FreeChunker motivate preserving useful retrieval representations
# at more than one granularity. Summary v2 is Cephalon's bounded, deterministic
# adaptation: patterns decide which original parent units deserve the small
# summary-node budget, but never invent or paraphrase evidence. The categories
# mirror common scientific retrieval needs and remain inspectable without
# adding a model call during ingestion.
SUMMARY_CATEGORY_PATTERNS = {
    "Definition": re.compile(
        r"\b(?:we define|is defined as|refers to|denotes|is an?|are (?:an?|the)|means)\b",
        re.IGNORECASE,
    ),
    "Method": re.compile(
        r"\b(?:we (?:use|used|propose|introduce|develop)|method|approach|pipeline|protocol|dataset|experiment)\b",
        re.IGNORECASE,
    ),
    "Result": re.compile(
        r"\b(?:we (?:find|found|show|demonstrate|observe|report)|results? show|achiev(?:e|ed)|"
        r"outperform(?:s|ed)?|improv(?:e|ed|ement)|increase(?:d)?|decrease(?:d)?|significant(?:ly)?)\b",
        re.IGNORECASE,
    ),
    "Limitation": re.compile(
        r"\b(?:limitation|caveat|however|although|despite|restricted to|may not|cannot|future work)\b",
        re.IGNORECASE,
    ),
    "Conclusion": re.compile(
        r"\b(?:we conclude|in conclusion|overall|therefore|thus|these findings|our findings)\b",
        re.IGNORECASE,
    ),
}
SUMMARY_NUMBER_PATTERN = re.compile(
    r"(?<!\w)[-+]?(?:\d+(?:,\d{3})*(?:\.\d+)?|\.\d+)"
    r"(?:[eE][-+]?\d+)?(?:\s?(?:%|ms|s|kg|g|mg|km|m|cm|mm|K|°C|Hz|GHz|V|W|Wh))?"
)
SUMMARY_ENTITY_PATTERN = re.compile(
    r"(?:"
    r"[A-Z][A-Za-z0-9+.-]*[A-Za-z0-9+]"
    r"(?:\s+[A-Z][A-Za-z0-9+.-]*[A-Za-z0-9+]){1,4}"
    r"|(?<![\w-])[A-Z][A-Z0-9]{1,}(?:-[A-Za-z0-9]+)*(?![\w-])"
    r"|(?<![\w-])[A-Za-z]*[A-Z][A-Za-z]*\d+[A-Za-z0-9-]*(?![\w-])"
    r")"
)
SUMMARY_ENTITY_STOPWORDS = {
    "Abstract",
    "Conclusion",
    "Discussion",
    "Figure",
    "Introduction",
    "Method",
    "Methods",
    "Result",
    "Results",
    "Section",
    "Table",
    "The",
    "This",
    "These",
}


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
    doc_id = existing_doc_id or str(uuid.uuid4())
    if not os.path.isfile(file_path):
        _restore_reindex_document_status(app_state, existing_doc_id, "Path is not a file.")
        return {"status": "failed", "path": file_path, "doc_id": doc_id, "error": "Path is not a file."}
    if not documents.collect_supported_files(file_path, force_text=force_text):
        _restore_reindex_document_status(app_state, existing_doc_id, "Unsupported file type.")
        return {"status": "failed", "path": file_path, "doc_id": doc_id, "error": "Unsupported file type."}

    persistence_started = False
    asset_transaction = None
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
            summary = summarize_parent(parent_text, parent.blocks)
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
        if os.path.splitext(file_path)[1].lower() == ".pdf":
            asset_transaction = await asyncio.to_thread(
                document_assets.AssetTransaction.prepare,
                app_state.settings.data_dir,
                doc_id,
                extracted.assets,
            )
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
                asset_rows=asset_transaction.rows if asset_transaction else [],
                before_commit=asset_transaction.promote if asset_transaction else None,
            )
        except Exception:
            if asset_transaction is not None:
                asset_transaction.rollback()
            if existing_doc_id:
                await asyncio.to_thread(_restore_document_vectors, app_state, doc_id, previous_vectors or [])
            else:
                await asyncio.to_thread(delete_document_vectors, app_state, doc_id)
            raise

        documents.mark_document_ready(app_state.sqlite, doc_id, child_count)
        if asset_transaction is not None:
            asset_transaction.finalize()
        await _report_progress(progress, "complete", 100)
        return {"status": "ready", "path": file_path, "doc_id": doc_id, "chunks": child_count, "extraction_mode": extraction_mode}
    except Exception as exc:
        if asset_transaction is not None:
            asset_transaction.rollback()
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


def _restore_reindex_document_status(app_state, doc_id: str | None, error: str) -> None:
    """Keep prior indexed documents visible when a source has disappeared."""
    if not doc_id:
        return
    previous_count = storage.fetchone(app_state.sqlite, "SELECT COUNT(*) AS count FROM chunks WHERE doc_id = ?", (doc_id,))
    storage.execute(
        app_state.sqlite,
        "UPDATE documents SET status = 'ready', chunk_count = ?, last_error = ? WHERE id = ?",
        (previous_count["count"] if previous_count else 0, error, doc_id),
    )


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
    asset_rows: list[tuple],
    before_commit: Callable[[], None] | None = None,
) -> None:
    with storage.SQLITE_LOCK:
        cursor = sqlite_conn.cursor()
        try:
            cursor.execute("BEGIN")
            cursor.execute("DELETE FROM chunks_fts WHERE doc_id = ?", (doc_id,))
            cursor.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
            cursor.execute("DELETE FROM summary_nodes WHERE doc_id = ?", (doc_id,))
            cursor.execute("DELETE FROM parent_chunks WHERE doc_id = ?", (doc_id,))
            cursor.execute("DELETE FROM document_assets WHERE doc_id = ?", (doc_id,))
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
            cursor.executemany(
                """
                INSERT INTO document_assets (
                    id, doc_id, page_number, bounding_box, filename, mime_type,
                    sha256, caption, width, height, size_bytes
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                asset_rows,
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
            if before_commit is not None:
                before_commit()
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


def _split_units_to_token_limit(units: list[str], maximum: int) -> list[str]:
    """Split boundary-less input deterministically without losing text."""
    bounded: list[str] = []
    for unit in units:
        words = unit.split()
        if len(words) <= maximum:
            bounded.append(unit)
            continue
        bounded.extend(" ".join(words[start:start + maximum]) for start in range(0, len(words), maximum))
    return bounded


def _chunking_config(settings: RagSettings) -> dict[str, int | str]:
    return {
        "parent_target_tokens": settings.parent_target_tokens,
        "parent_max_tokens": settings.parent_max_tokens,
        "child_target_tokens": settings.child_target_tokens,
        "child_max_tokens": settings.child_max_tokens,
        "child_overlap_tokens": settings.child_overlap_tokens,
        # Summary vectors are stored in the same LanceDB table as child
        # vectors. Changing their selection policy must therefore mark every
        # prior document stale and require an explicit reindex.
        "summary_profile": SUMMARY_PROFILE,
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
            "element_ids": [
                block.element_id
                for block in blocks
                if block.element_id
            ],
            "asset_ids": list(dict.fromkeys(
                asset_id
                for block in blocks
                for asset_id in block.asset_ids
            )),
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
    # JSON, HTML, CSVs, and malformed PDF fallbacks can contain a single
    # enormous line with no sentence boundary. Bound it before batching so a
    # malformed unit cannot make llama.cpp reject the entire document.
    units = _split_units_to_token_limit(split_text_units(text), maximum)
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
    units = _split_units_to_token_limit(split_text_units(parent_text), maximum)
    if not units:
        return []
    # Keep child chunks aligned to sentence/paragraph units. Embeddings are
    # created only once during the final batched indexing pass.
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


def summarize_parent(text: str, blocks: list[DocumentBlock] | None = None) -> str:
    """Build a bounded, extractive scientific summary for dense retrieval.

    The previous implementation copied the first three units, which strongly
    favored introductions and often omitted the measured result or caveat.
    Version 2 allocates the same 700-character storage budget across headings,
    repeated entities, definitions, methods, results, numbers, limitations,
    and conclusions. Text is only whitespace-normalized and boundary-truncated;
    it is never generated or paraphrased, preserving the evidence/provenance
    boundary.

    Complexity is O(parent characters + units × categories). Parents are
    already capped at 650 tokens, and both emitted units and entities have hard
    limits, so this cannot create an unbounded ingestion-time loop.
    """

    units = [_normalize_summary_unit(unit) for unit in split_text_units(text)]
    units = [unit for unit in units if unit]
    if not units:
        return _truncate_summary(text.strip(), SUMMARY_MAX_CHARACTERS)
    if len(units) == 1 and not _summary_heading_path(blocks or []):
        # A short one-unit parent already is its own lossless summary. Labels
        # would add storage and alter rollback fixtures without improving the
        # retrieval representation.
        return _truncate_summary(units[0], SUMMARY_MAX_CHARACTERS)

    components: list[str] = []
    heading_path = _summary_heading_path(blocks or [])
    if heading_path:
        heading = _truncate_summary(
            " > ".join(heading_path),
            SUMMARY_MAX_METADATA_CHARACTERS,
        )
        components.append(f"Section: {heading}")

    entities = _summary_entities(text)
    if entities:
        entity_text = _truncate_summary(
            "; ".join(entities),
            SUMMARY_MAX_METADATA_CHARACTERS,
        )
        components.append(f"Entities: {entity_text}")

    numbers = _summary_numbers(text)
    if numbers:
        number_text = _truncate_summary(
            "; ".join(numbers),
            SUMMARY_MAX_METADATA_CHARACTERS,
        )
        components.append(f"Values: {number_text}")

    selected_indexes: set[int] = set()
    selected_units: list[tuple[int, str, str]] = []
    category_candidates: dict[str, tuple[int, str]] = {}
    for label, pattern in SUMMARY_CATEGORY_PATTERNS.items():
        candidate = _best_summary_unit(units, pattern, label, selected_indexes)
        if candidate is not None:
            selected_indexes.add(candidate)
            category_candidates[label] = (candidate, units[candidate])

    # Results carry the highest retrieval value for evidence-dense scientific
    # questions. Definitions and methods follow, while caveats and conclusions
    # remain eligible when they have an unclaimed source unit.
    for label in ("Result", "Definition", "Method", "Limitation", "Conclusion"):
        if label in category_candidates:
            index, unit = category_candidates[label]
            selected_units.append((index, label, unit))

    if not selected_units:
        selected_units.append((0, "Context", units[0]))
        selected_indexes.add(0)

    # Fill remaining slots with high-information context units after every
    # detected evidence category has had an opportunity to claim a slot.
    remaining = sorted(
        (
            (_summary_unit_score(unit, index, len(units)), index, unit)
            for index, unit in enumerate(units)
            if index not in selected_indexes
        ),
        reverse=True,
    )
    while len(selected_units) < SUMMARY_MAX_CONTENT_UNITS and remaining:
        _score, index, unit = remaining.pop(0)
        selected_units.append((index, "Context", unit))

    selected_units = selected_units[:SUMMARY_MAX_CONTENT_UNITS]
    label_overhead = sum(len(label) + 2 for _index, label, _unit in selected_units)
    separator_overhead = len(components) + len(selected_units)
    available_for_units = max(
        SUMMARY_MIN_UNIT_CHARACTERS * len(selected_units),
        SUMMARY_MAX_CHARACTERS
        - len(" ".join(components))
        - label_overhead
        - separator_overhead,
    )
    per_unit_budget = min(
        SUMMARY_MAX_UNIT_CHARACTERS,
        max(
            SUMMARY_MIN_UNIT_CHARACTERS,
            available_for_units // max(1, len(selected_units)),
        ),
    )
    components.extend(
        f"{label}: {_truncate_summary(unit, per_unit_budget)}"
        for _index, label, unit in selected_units
    )
    return _truncate_summary(" ".join(components), SUMMARY_MAX_CHARACTERS)


def _normalize_summary_unit(text: str) -> str:
    """Collapse extraction whitespace without changing lexical content."""

    return re.sub(r"\s+", " ", text).strip()


def _summary_heading_path(blocks: list[DocumentBlock]) -> list[str]:
    """Return the first stable structured heading path for this parent."""

    for block in blocks:
        if block.heading_path:
            return [
                _normalize_summary_unit(value)
                for value in block.heading_path
                if _normalize_summary_unit(value)
            ]
    return []


def _summary_entities(text: str) -> list[str]:
    """Select repeated or distinctive entity strings in deterministic order."""

    counts: dict[str, tuple[int, int]] = {}
    for match in SUMMARY_ENTITY_PATTERN.finditer(text):
        entity = _normalize_summary_unit(match.group(0))
        if entity in SUMMARY_ENTITY_STOPWORDS or len(entity) < 2:
            continue
        count, first_position = counts.get(entity, (0, match.start()))
        counts[entity] = (count + 1, first_position)
    ranked = sorted(
        counts,
        key=lambda entity: (
            -counts[entity][0],
            counts[entity][1],
            entity.casefold(),
        ),
    )
    return ranked[:SUMMARY_MAX_ENTITIES]


def _summary_numbers(text: str) -> list[str]:
    """Return unique values with units in source order for numeric retrieval."""

    values: list[str] = []
    seen: set[str] = set()
    for match in SUMMARY_NUMBER_PATTERN.finditer(text):
        value = _normalize_summary_unit(match.group(0))
        normalized = value.casefold()
        if normalized in seen:
            continue
        seen.add(normalized)
        values.append(value)
        if len(values) >= 8:
            break
    return values


def _best_summary_unit(
    units: list[str],
    pattern: re.Pattern[str],
    label: str,
    excluded: set[int],
) -> int | None:
    """Choose the strongest unselected exact unit for one evidence category."""

    candidates: list[tuple[float, int]] = []
    for index, unit in enumerate(units):
        if index in excluded or not pattern.search(unit):
            continue
        score = _summary_unit_score(unit, index, len(units))
        score += 4.0
        if label in {"Result", "Numbers"} and SUMMARY_NUMBER_PATTERN.search(unit):
            score += 2.0
        if label in {"Limitation", "Conclusion"}:
            # Caveats and conclusions commonly occur late in a section; a
            # bounded position bonus offsets the general opening-unit bias.
            score += index / max(1, len(units) - 1)
        candidates.append((score, index))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (-item[0], item[1]))
    return candidates[0][1]


def _summary_unit_score(unit: str, index: int, unit_count: int) -> float:
    """Score information density without relying on corpus-specific terms."""

    words = unit.split()
    entity_count = len(SUMMARY_ENTITY_PATTERN.findall(unit))
    number_count = len(SUMMARY_NUMBER_PATTERN.findall(unit))
    category_count = sum(
        1 for pattern in SUMMARY_CATEGORY_PATTERNS.values() if pattern.search(unit)
    )
    length_score = min(len(words), 60) / 60
    opening_bonus = 0.5 if index == 0 else 0.0
    closing_bonus = 0.35 if index == unit_count - 1 else 0.0
    return (
        entity_count * 0.6
        + number_count * 0.8
        + category_count
        + length_score
        + opening_bonus
        + closing_bonus
    )


def _truncate_summary(text: str, maximum: int) -> str:
    """Truncate on a word boundary while keeping the hard storage limit."""

    clean = _normalize_summary_unit(text)
    if len(clean) <= maximum:
        return clean
    boundary = clean.rfind(" ", 0, maximum + 1)
    if boundary <= 0:
        boundary = maximum
    return clean[:boundary].rstrip(" ,;:-")


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
