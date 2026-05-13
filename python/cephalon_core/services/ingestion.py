import os
import time

from langchain_text_splitters import RecursiveCharacterTextSplitter

from .. import storage
from ..schemas import RagSettings
from . import documents
from .retrieval import ensure_vector_table, get_embedding, vector_table_name


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
        storage.delete_document_fts(app_state.sqlite, doc_id)
        storage.execute(app_state.sqlite, "DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
        if not raw_text.strip():
            raise ValueError("No extractable text found.")

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=rag_settings.chunk_size,
            chunk_overlap=rag_settings.chunk_overlap,
            separators=["\n\n", "\n", r"(?<=\. )", " ", ""],
        )
        chunks = splitter.split_text(raw_text)
        if not chunks:
            raise ValueError("No text chunks produced.")

        lance_data = []
        for index, text in enumerate(chunks):
            chunk_id = f"{doc_id}_{index}"
            vector = await get_embedding(app_state, text)
            storage.execute(
                app_state.sqlite,
                """
                INSERT INTO chunks (id, doc_id, chunk_index, text, chunk_length, embedding_model_id, embedding_dim)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    chunk_id,
                    doc_id,
                    index,
                    text,
                    len(text),
                    metadata["embedding_model_id"],
                    metadata["embedding_dim"],
                ),
            )
            storage.upsert_chunk_fts(app_state.sqlite, chunk_id, doc_id, text)
            lance_data.append({
                "vector": vector,
                "id": chunk_id,
                "doc_id": doc_id,
                "text": text,
                "chunk_index": index,
                **metadata,
                "chunk_length": len(text),
            })

        ensure_vector_table(app_state, lance_data)

        documents.mark_document_ready(app_state.sqlite, doc_id, len(chunks))
        return {"status": "ready", "path": file_path, "doc_id": doc_id, "chunks": len(chunks), "extraction_mode": extraction_mode}
    except Exception as exc:
        if doc_id:
            documents.mark_document_failed(app_state.sqlite, doc_id, str(exc))
        return {"status": "failed", "path": file_path, "doc_id": doc_id, "error": str(exc)}


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
        cursor.execute("DELETE FROM document_tags WHERE doc_id = ?", (doc_id,))
        cursor.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
        app_state.sqlite.commit()


def mark_reindexing(app_state, doc_id: str) -> str:
    row = storage.fetchone(app_state.sqlite, "SELECT path FROM documents WHERE id = ? AND type = 'file'", (doc_id,))
    if not row:
        raise ValueError("Document not found.")
    storage.execute(app_state.sqlite, "UPDATE documents SET status = 'queued', last_error = NULL WHERE id = ?", (doc_id,))
    return row["path"]
