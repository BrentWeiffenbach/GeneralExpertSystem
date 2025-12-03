from __future__ import annotations

from typing import Any, Dict, List, Optional

from gemini import GeminiAPI
from .document_processor import DocumentProcessor
from .vector_store import Chunk, VectorStore

USER_ASSISTANT_PROMPT = """You are an Expert Process Assistant, an AI that combines uploaded manuals with field notes.
Answer with clear, actionable steps. Cite specific codes, procedures, or rules when available.
Prefer concrete instructions over speculation. If the knowledge base lacks an answer, say so and ask for more detail."""


class RAGEngine:
    """Coordinates retrieval augmented responses and learning hooks."""

    def __init__(
        self,
        gemini: GeminiAPI,
        vector_store: VectorStore,
        document_processor: DocumentProcessor,
    ) -> None:
        self.gemini = gemini
        self.vector_store = vector_store
        self.document_processor = document_processor

    def answer(self, messages: List[Dict[str, str]], top_k: int = 4) -> Dict[str, Any]:
        if not messages:
            raise ValueError("No conversation supplied")

        latest_question = messages[-1]["content"]
        retrieved = self.vector_store.search(latest_question, top_k=top_k)
        context_blob = self._format_context(retrieved)

        prompt_messages: List[Dict[str, Any]] = [
            {
                "role": "user",
                "parts": [
                    {
                        "text": (
                            "Context snippets:\n"
                            + context_blob
                            + "\nUse the context only when relevant."
                        )
                    }
                ],
            }
        ]
        prompt_messages.extend(self._to_parts(messages))

        response = self.gemini.call(prompt_messages, system_prompt=USER_ASSISTANT_PROMPT)
        answer_text = self.gemini.extract_text(response).strip()

        return {
            "answer": answer_text,
            "context": [self._chunk_to_dict(chunk) for chunk in retrieved],
        }

    def maybe_learn(self, conversation: List[Dict[str, str]], source_label: str) -> Optional[Dict[str, Any]]:
        if len(conversation) < 4:
            return None
        if conversation[-1].get("role") != "model":
            return None
        return self.document_processor.summarize_conversation(conversation, source_label)

    def _format_context(self, chunks: List[Chunk]) -> str:
        if not chunks:
            return "(no matching chunks yet)"
        lines = []
        for chunk in chunks:
            source_display = chunk.source
            if chunk.metadata and chunk.metadata.get("download_link"):
                source_display = f"[{chunk.source}]({chunk.metadata['download_link']})"
            
            lines.append(
                f"[{chunk.id}] {chunk.title} — Source: {source_display}\n{chunk.content}"
            )
        return "\n\n".join(lines)

    def _chunk_to_dict(self, chunk: Chunk) -> Dict[str, Any]:
        data = {
            "id": chunk.id,
            "title": chunk.title,
            "source": chunk.source,
            "content": chunk.content,
            "created_at": chunk.created_at,
            "metadata": chunk.metadata,
            "chunk_index": chunk.chunk_index,
        }
        if chunk.score is not None:
            data["score"] = round(chunk.score, 3)
        return data

    def _to_parts(self, messages: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        formatted: List[Dict[str, Any]] = []
        for message in messages[-15:]:
            formatted.append(
                {
                    "role": message.get("role", "user"),
                    "parts": [{"text": message.get("content", "")}],
                }
            )
        return formatted
