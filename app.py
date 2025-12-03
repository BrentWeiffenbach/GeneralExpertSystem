from __future__ import annotations

import os
import shutil
import time
import traceback
import uuid
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv
from flask import Flask, jsonify, redirect, render_template, request, url_for, send_from_directory
from werkzeug.utils import secure_filename

from gemini import GeminiAPI
from ges_core.document_processor import DocumentProcessor
from ges_core.rag import RAGEngine
from ges_core.vector_store import VectorStore

BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "uploads"
DATA_DIR = BASE_DIR / "data"
BACKUP_DIR = DATA_DIR / "backups"
DB_PATH = DATA_DIR / "knowledge.sqlite"

UPLOAD_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)
BACKUP_DIR.mkdir(exist_ok=True)

EXPERT_ASSISTANT_PROMPT = """You are an expert knowledge capture assistant.
The user is a field expert providing new information, troubleshooting steps, or general insights.
Your goal is to ensure the information is complete, actionable, and correctly integrated.

1. **Context Awareness**: You will receive "Existing Knowledge Base Context". Always check this first.
2. **Conflict Resolution**:
   - If the user's input conflicts with existing context (e.g., changing a price, procedure, or rule), DO NOT assume it's a simple update.
   - **Ask for Clarification**: "I see we currently have [Existing Value] for [Topic]. Is your intent to REPLACE this entirely, UPDATE specific details (e.g. just the price), or ADD this as a new variation/edge case?"
   - This is critical to prevent accidental data loss (e.g. overwriting a whole table when only one row changed).
3. **Completeness**: If details are missing (e.g. "The server is down" -> "Which server? What error code?"), ask for them.
4. **Finalization**: When you have gathered sufficient details and clarified the intent (Update vs Replace vs Add), explicitly ask the expert to click the 'Save' button to commit this knowledge.
5. Be concise (under 6 sentences)."""

app = Flask(__name__)

load_dotenv()

gemini_client = GeminiAPI()
vector_store = VectorStore(DB_PATH, embedder=gemini_client.embed)
document_processor = DocumentProcessor(gemini_client, vector_store)
rag_engine = RAGEngine(gemini_client, vector_store, document_processor)


def _error_response(message: str, status: int = 400):
    return jsonify({"ok": False, "error": message}), status


@app.route("/")
def root() -> Any:
    return redirect(url_for("user_view"))


@app.route("/user")
def user_view() -> Any:
    return render_template("user.html")


@app.route("/admin")
def admin_view() -> Any:
    return render_template("admin.html")


@app.route("/admin/debug")
def admin_debug_view() -> Any:
    return render_template("admin_debug.html")


@app.route("/uploads/<path:filename>")
def serve_upload(filename: str) -> Any:
    return send_from_directory(UPLOAD_DIR, filename)


@app.post("/api/admin/upload")
def upload_document() -> Any:
    file = request.files.get("file")
    if not file or file.filename == "":
        return _error_response("Please choose a PDF, CSV, or TXT file", 422)

    filename = secure_filename(file.filename or "upload")
    suffix = Path(filename).suffix.lower()
    if suffix not in {".pdf", ".csv", ".txt"}:
        return _error_response("Only PDF, CSV, and TXT files are supported", 422)

    saved_name = f"{uuid.uuid4().hex}_{filename}"
    saved_path = UPLOAD_DIR / saved_name
    file.save(saved_path)

    try:
        result = document_processor.ingest_document(saved_path, filename, saved_name=saved_name)
        return jsonify({"ok": True, "summary": result})
    except Exception as exc:  # pragma: no cover - surfaced to UI
        traceback.print_exc()
        return _error_response(str(exc), 500)


@app.post("/api/chat/user")
def user_chat() -> Any:
    payload = request.get_json(force=True)
    messages = _sanitize_messages(payload.get("messages", []))
    if not messages:
        return _error_response("Conversation history missing", 422)

    try:
        rag_result = rag_engine.answer(messages)
        # User sessions are read-only now; no learning from them.
        return jsonify(
            {
                "ok": True,
                "reply": rag_result["answer"],
                "context": rag_result["context"],
            }
        )
    except Exception as exc:  # pragma: no cover
        traceback.print_exc()
        return _error_response(str(exc), 500)


@app.post("/api/chat/expert")
def expert_chat() -> Any:
    payload = request.get_json(force=True)
    messages = _sanitize_messages(payload.get("messages", []))
    if not messages:
        return _error_response("Conversation history missing", 422)

    try:
        # Retrieve context to inform the expert chat
        # Use the last 2 user messages to form a better search query
        user_msgs = [m["content"] for m in messages if m["role"] == "user"]
        search_query = " ".join(user_msgs[-2:]) if user_msgs else ""
        
        context_chunks = vector_store.search(search_query, top_k=3) if search_query else []
        
        context_text = ""
        if context_chunks:
            context_text = "\nExisting Knowledge Base Context:\n" + "\n".join(
                [f"[{c.id}] {c.title}: {c.content}" for c in context_chunks]
            )

        # Inject context into the last message for the model to see
        prompt_messages = messages[-15:]
        if context_text and prompt_messages:
             # We append context to the last user message effectively
             last_msg = prompt_messages[-1]
             if last_msg["role"] == "user":
                 last_msg["content"] += f"\n\n{context_text}"

        gemini_response = gemini_client.call(prompt_messages, system_prompt=EXPERT_ASSISTANT_PROMPT)
        reply_text = gemini_client.extract_text(gemini_response).strip()
        
        return jsonify(
            {
                "ok": True,
                "reply": reply_text,
                "autoChunk": None,
            }
        )
    except Exception as exc:  # pragma: no cover
        traceback.print_exc()
        return _error_response(str(exc), 500)


@app.post("/api/expert/save")
def expert_save() -> Any:
    payload = request.get_json(force=True)
    messages = _sanitize_messages(payload.get("messages", []))
    if not messages:
        return _error_response("Conversation history missing", 422)

    try:
        chunk = document_processor.summarize_conversation(messages, "expert-manual", force=True)
        if not chunk:
             return _error_response("No actionable content found to save.", 400)
        return jsonify({"ok": True, "chunk": chunk})
    except Exception as exc:  # pragma: no cover
        traceback.print_exc()
        return _error_response(str(exc), 500)


@app.get("/api/chunks")
def list_chunks() -> Any:
    limit = int(request.args.get("limit", 200))
    chunks = vector_store.list_chunks(limit=limit)
    total = vector_store.count()
    return jsonify(
        {
            "ok": True,
            "total": total,
            "chunks": [
                {
                    "id": chunk.id,
                    "title": chunk.title,
                    "source": chunk.source,
                    "content": chunk.content,
                    "metadata": chunk.metadata,
                    "created_at": chunk.created_at,
                    "chunk_index": chunk.chunk_index,
                }
                for chunk in chunks
            ],
        }
    )


@app.delete("/api/chunks/<int:chunk_id>")
def delete_chunk(chunk_id: int) -> Any:
    try:
        success = vector_store.delete_chunk(chunk_id)
        if not success:
            return _error_response("Chunk not found", 404)
        return jsonify({"ok": True})
    except Exception as exc:
        return _error_response(str(exc), 500)


@app.put("/api/chunks/<int:chunk_id>")
def update_chunk(chunk_id: int) -> Any:
    try:
        payload = request.get_json(force=True)
        content = payload.get("content")
        metadata = payload.get("metadata")
        
        if not content:
            return _error_response("Content is required", 422)
            
        success = vector_store.update_chunk(chunk_id, content, metadata)
        if not success:
            return _error_response("Chunk not found", 404)
            
        return jsonify({"ok": True})
    except Exception as exc:
        return _error_response(str(exc), 500)


@app.post("/api/admin/reset")
def reset_db() -> Any:
    try:
        vector_store.reset()
        return jsonify({"ok": True})
    except Exception as exc:
        return _error_response(str(exc), 500)


@app.post("/api/admin/backup")
def backup_db() -> Any:
    try:
        payload = request.get_json(silent=True) or {}
        name = payload.get("name")
        if not name:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            name = f"backup-{timestamp}"

        name = secure_filename(name)
        if not name.endswith(".sqlite"):
            name += ".sqlite"

        dest = BACKUP_DIR / name
        shutil.copy2(DB_PATH, dest)
        return jsonify({"ok": True, "name": name})
    except Exception as exc:
        return _error_response(str(exc), 500)


@app.get("/api/admin/backups")
def list_backups() -> Any:
    backups = []
    for f in BACKUP_DIR.glob("*.sqlite"):
        backups.append({
            "name": f.name,
            "created": time.ctime(f.stat().st_ctime),
            "size": f.stat().st_size
        })
    backups.sort(key=lambda x: x["created"], reverse=True)
    return jsonify({"ok": True, "backups": backups})


@app.post("/api/admin/restore")
def restore_db() -> Any:
    try:
        payload = request.get_json(force=True)
        name = payload.get("name")
        if not name:
            return _error_response("Backup name required", 422)

        src = BACKUP_DIR / secure_filename(name)
        if not src.exists():
            return _error_response("Backup not found", 404)

        shutil.copy2(src, DB_PATH)
        return jsonify({"ok": True})
    except Exception as exc:
        return _error_response(str(exc), 500)


def _sanitize_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    clean: List[Dict[str, str]] = []
    for message in messages:
        role = message.get("role", "user")
        content = str(message.get("content", "")).strip()
        if not content:
            continue
        clean.append({"role": role, "content": content})
    return clean


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
