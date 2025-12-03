from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from pypdf import PdfReader

from gemini import GeminiAPI
from .vector_store import VectorStore


CHUNK_ANALYZER_PROMPT = """You are a troubleshooting knowledge architect. Given a snippet of text from a manual, log, or support document, pull out the key metadata.
Respond ONLY in JSON with this shape:
{
  "section_title": string,
  "error_codes": [string],
  "problems": [string],
  "solutions": [string],
  "procedures": [string],
  "tags": [string]
}
If a field is unknown, emit an empty array or empty string. Keep titles under 80 characters."""

CONVERSATION_TO_CHUNK_PROMPT = """You are an expert system that captures knowledge from conversations.
You will receive a transcript and potentially some existing context chunks.
Your task is to summarize the FINAL CONSENSUS or NEW KNOWLEDGE from the conversation into a concise knowledge chunk.
Decide if we should ADD new knowledge or UPDATE an existing chunk.
Respond in JSON:
{
  "action": "create" | "update" | "none",
  "target_chunk_id": number | null,
  "title": string,
  "content": string,
  "tags": [string]
}
Use "update" if the new information is a modification, correction, or update to an existing item found in the context chunks (e.g. updating a price in a list, changing a status).
Use "create" ONLY if the information is completely new and unrelated to the existing context chunks.
Use "none" ONLY if the conversation is purely chit-chat with no technical or procedural value.
If the user explicitly asks to save, you MUST extract the best possible summary in "content" and set action to "create" (or "update").
"content" must be a standalone summary of the knowledge, not just a transcript.
If action is "update":
1. "content" must be the NEW, COMPLETE content for that chunk (replacing the old content).
2. If the chunk is a list or table (e.g. a price list), you MUST rewrite the ENTIRE chunk with the specific item updated, keeping all other items exactly as they were. Do not truncate the other items.
3. "target_chunk_id" must be the ID of the chunk being updated."""

UPDATE_VERIFICATION_PROMPT = """You are a data integrity guardian.
Your task is to compare an ORIGINAL text with a PROPOSED UPDATE to ensure no valuable information was accidentally lost.
The user intended to make a specific change (e.g. update a price, add a note).
However, sometimes the update process accidentally deletes unrelated information (e.g. other items in a list, other sections).
It could also remove generalized information for specific information, but it is important to keep BOTH.

Original Text:
{original_text}

Proposed Update:
{new_text}

User's Intent (inferred):
{user_intent}

Task:
1. Check if any information present in the Original Text is missing from the Proposed Update.
2. Determine if that missing information was *intended* to be removed based on the User's Intent.
3. If valuable information was lost accidentally, reconstruct the text to include BOTH the update AND the missing original information.
4. If the Proposed Update is safe (no accidental loss), return it as is.

Respond in JSON:
{{
  "safe": boolean,
  "reasoning": string,
  "corrected_content": string
}}
"safe": true if no accidental data loss occurred.
"reasoning": Explain what changed and if anything was preserved or lost.
"corrected_content": The final text to use. If safe, this matches Proposed Update. If unsafe, this is the fixed version."""


class DocumentProcessor:
    """Handles ingestion, chunk annotation, and conversation summaries."""

    def __init__(self, gemini: GeminiAPI, vector_store: VectorStore) -> None:
        self.gemini = gemini
        self.vector_store = vector_store

    def ingest_document(self, file_path: Path, original_name: str, saved_name: Optional[str] = None, max_chunks: int = 40) -> Dict[str, Any]:
        text = self._extract_text(file_path)
        chunks = self._chunk_text(text)
        chunk_summaries: List[Dict[str, Any]] = []

        download_link = f"/uploads/{saved_name}" if saved_name else None
        current_page = 1

        for idx, chunk_text in enumerate(chunks[:max_chunks]):
            # Extract page number if present
            page_matches = re.findall(r"\[\[PAGE_(\d+)\]\]", chunk_text)
            if page_matches:
                current_page = int(page_matches[0])
            
            # Remove page markers from the content we store/analyze
            clean_content = re.sub(r"\[\[PAGE_\d+\]\]\s*", "", chunk_text).strip()
            if not clean_content:
                continue

            metadata = self._analyze_chunk(clean_content)
            title = metadata.get("section_title") or f"{original_name} chunk {idx + 1}"
            enriched_metadata = {
                "type": "document",
                "chunk_index": idx,
                "download_link": download_link,
                "page_number": current_page,
                **metadata
            }
            chunk_id = self.vector_store.add_chunk(
                title=title,
                content=clean_content,
                source=original_name,
                metadata=enriched_metadata,
                chunk_index=idx,
            )
            chunk_summaries.append(
                {
                    "chunk_id": chunk_id,
                    "title": title,
                    "metadata": enriched_metadata,
                }
            )

        return {
            "total_chunks": len(chunk_summaries),
            "chunks": chunk_summaries,
        }

    def summarize_conversation(
        self,
        messages: List[Dict[str, str]],
        source_label: str,
        force: bool = False,
    ) -> Optional[Dict[str, Any]]:
        transcript = self._format_conversation(messages)
        
        # Check for existing context to see if we should update
        # Use the last few user messages to form a robust search query
        user_msgs = [m["content"] for m in messages if m["role"] == "user"]
        # Filter out very short messages (e.g. "Save", "Yes") to find the actual topic
        meaningful_msgs = [m for m in user_msgs if len(m) > 10]
        search_query = " ".join(meaningful_msgs[-3:]) if meaningful_msgs else " ".join(user_msgs[-3:])
        
        context_chunks = self.vector_store.search(search_query, top_k=5) if search_query else []
        
        context_text = ""
        if context_chunks:
            # Provide more content context so the model can see what it's updating
            context_text = "\nExisting Knowledge Base Context (ID: Title - Content):\n" + "\n".join(
                [f"ID {c.id}: {c.title}\nContent: {c.content}" for c in context_chunks]
            )

        prompt_text = f"Transcript:\n{transcript}\n{context_text}"
        if force:
            prompt_text += "\n\nIMPORTANT: The user has explicitly requested to SAVE this conversation. You MUST extract the final consensus or information as a chunk. If the user is correcting an existing chunk shown in context, use action='update' and the correct ID. If it is new, use 'create'. Content must be a detailed summary."

        response = self.gemini.call(
            [
                {
                    "role": "user",
                    "parts": [
                        {
                            "text": prompt_text
                        }
                    ],
                }
            ],
            system_prompt=CONVERSATION_TO_CHUNK_PROMPT,
        )
        raw_response = self.gemini.extract_text(response)
        parsed = self._safe_json(raw_response)
        
        action = parsed.get("action")
        if action == "none":
            if force:
                action = "create" # Force create if user insisted
            else:
                return None
            
        # Legacy fallback
        if "should_add" in parsed and parsed["should_add"]:
            action = "create"

        title = parsed.get("title") or f"Field insight ({source_label})"
        content = parsed.get("content") or ""
        
        # Fallback for legacy or hallucinated fields
        if not content:
            prob = parsed.get("problem", "")
            sol = parsed.get("solution", "")
            if prob or sol:
                content = f"Problem: {prob}\nSolution: {sol}"

        if not content:
            if force:
                # Fallback: Use the last user message as content
                last_user_content = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "No content provided.")
                content = f"Expert Update: {last_user_content}"
                title = title or "Expert Update"
            else:
                return None

        metadata = {
            "type": source_label,
            "tags": parsed.get("tags", []),
        }

        if action == "update" and parsed.get("target_chunk_id"):
            chunk_id = int(parsed["target_chunk_id"])
            
            # Verification Step
            original_chunk = self.vector_store.get_chunk(chunk_id)
            verification_note = ""
            if original_chunk:
                print(f"DEBUG: Verifying update for chunk {chunk_id}")
                user_intent = " ".join([m["content"] for m in messages if m["role"] == "user"][-3:])
                verified = self._verify_update(original_chunk.content, content, user_intent)
                if verified:
                    content = verified.get("corrected_content", content)
                    verification_note = verified.get("reasoning", "")
                    if not verified.get("safe"):
                        print(f"DEBUG: Data loss detected! Corrected content used. Reasoning: {verification_note}")

            self.vector_store.update_chunk(chunk_id, content, metadata)
            return {
                "chunk_id": chunk_id,
                "title": title,
                "action": "updated",
                "metadata": metadata,
                "verification_note": verification_note
            }
        
        # Default to create
        chunk_id = self.vector_store.add_chunk(
            title=title,
            content=content,
            source=source_label,
            metadata=metadata,
        )
        metadata["chunk_id"] = chunk_id
        metadata["title"] = title
        metadata["action"] = "created"
        return metadata

    def _verify_update(self, original_text: str, new_text: str, user_intent: str) -> Optional[Dict[str, Any]]:
        prompt = UPDATE_VERIFICATION_PROMPT.format(
            original_text=original_text,
            new_text=new_text,
            user_intent=user_intent
        )
        response = self.gemini.call(
            [{"role": "user", "parts": [{"text": prompt}]}]
        )
        return self._safe_json(self.gemini.extract_text(response))

    def _extract_text(self, file_path: Path) -> str:
        suffix = file_path.suffix.lower()
        if suffix == ".pdf":
            return self._extract_pdf(file_path)
        if suffix == ".csv":
            return self._extract_csv(file_path)
        if suffix == ".txt":
            return self._extract_txt(file_path)
        raise ValueError(f"Unsupported file type: {suffix}")

    def _extract_txt(self, file_path: Path) -> str:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()

    def _extract_pdf(self, file_path: Path) -> str:
        reader = PdfReader(str(file_path))
        pages = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            # Prepend page marker to track location
            pages.append(f"[[PAGE_{i+1}]]\n{text}")
        return "\n".join(pages)

    def _extract_csv(self, file_path: Path) -> str:
        lines: List[str] = []
        with open(file_path, "r", encoding="utf-8", newline="") as csv_file:
            reader = csv.reader(csv_file)
            for row in reader:
                lines.append(", ".join(cell.strip() for cell in row))
        return "\n".join(lines)

    def _chunk_text(self, text: str, max_chars: int = 900, overlap: int = 0) -> List[str]:
        # Split by double newlines to preserve paragraphs/sections
        paragraphs = re.split(r"\n\s*\n", text)
        chunks: List[str] = []
        buffer = ""

        for paragraph in paragraphs:
            # Normalize whitespace within the paragraph but keep it as a block
            clean_para = re.sub(r"\s+", " ", paragraph).strip()
            if not clean_para:
                continue
                
            if len(buffer) + len(clean_para) <= max_chars:
                buffer += ("\n\n" if buffer else "") + clean_para
            else:
                # If the paragraph itself is too huge, we might need to split it further
                if len(clean_para) > max_chars:
                     # Split huge paragraph by sentences
                     sentences = re.split(r"(?<=[.!?]) +", clean_para)
                     for sentence in sentences:
                         if len(buffer) + len(sentence) <= max_chars:
                             buffer += (" " if buffer else "") + sentence
                         else:
                             if buffer:
                                 chunks.append(buffer.strip())
                             buffer = sentence
                else:
                    if buffer:
                        chunks.append(buffer.strip())
                    buffer = clean_para

        if buffer:
            chunks.append(buffer.strip())

        if overlap and len(chunks) > 1:
            overlapped: List[str] = []
            prev_tail = ""
            for chunk in chunks:
                # Simple overlap strategy
                candidate = (prev_tail + "\n...\n" + chunk).strip() if prev_tail else chunk
                overlapped.append(candidate)
                # Keep the last bit of this chunk for the next one
                prev_tail = chunk[-overlap:]
            return overlapped

        return chunks

    def _analyze_chunk(self, chunk_text: str) -> Dict[str, Any]:
        response = self.gemini.call(
            [
                {
                    "role": "user",
                    "parts": [{"text": chunk_text}],
                }
            ],
            system_prompt=CHUNK_ANALYZER_PROMPT,
        )
        parsed = self._safe_json(self.gemini.extract_text(response))
        return parsed

    def _format_conversation(self, messages: List[Dict[str, str]], limit: int = 4000) -> str:
        lines = []
        for message in messages[-15:]:
            role = message.get("role", "user").upper()
            content = message.get("content", "").strip()
            lines.append(f"{role}: {content}")
        transcript = "\n".join(lines)
        return transcript[-limit:]

    def _safe_json(self, payload: str) -> Dict[str, Any]:
        # Strip markdown code blocks if present
        clean = payload.strip()
        if clean.startswith("```"):
            clean = clean.split("\n", 1)[1]
            if clean.endswith("```"):
                clean = clean.rsplit("\n", 1)[0]
        
        # Also handle the case where the first line is ```json
        clean = clean.replace("```json", "").replace("```", "").strip()

        try:
            return json.loads(clean)
        except json.JSONDecodeError:
            # Try to find the first { and last }
            start = clean.find("{")
            end = clean.rfind("}")
            if start != -1 and end != -1:
                try:
                    return json.loads(clean[start : end + 1])
                except json.JSONDecodeError:
                    pass
            return {}
