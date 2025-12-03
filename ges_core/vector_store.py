from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np


@dataclass
class Chunk:
    id: int
    title: str
    source: str
    content: str
    metadata: Dict[str, Any]
    created_at: str
    chunk_index: int
    score: Optional[float] = None


class VectorStore:
    def __init__(self, db_path: Path, embedder: Callable[[str], List[float]]) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.embedder = embedder
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    title TEXT,
                    source TEXT,
                    content TEXT,
                    metadata TEXT,
                    embedding TEXT,
                    created_at TEXT,
                    chunk_index INTEGER DEFAULT 0
                )
                """
            )
            self._ensure_chunk_index_column(conn)
            conn.commit()

    def _ensure_chunk_index_column(self, conn: sqlite3.Connection) -> None:
        columns = {row["name"] for row in conn.execute("PRAGMA table_info(chunks)")}
        if "chunk_index" not in columns:
            conn.execute("ALTER TABLE chunks ADD COLUMN chunk_index INTEGER DEFAULT 0")

    def add_chunk(
        self,
        *,
        title: str,
        content: str,
        source: str,
        metadata: Optional[Dict[str, Any]] = None,
        chunk_index: Optional[int] = None,
    ) -> int:
        embedding = self.embedder(content)
        serialized_metadata = json.dumps(metadata or {})
        serialized_embedding = json.dumps(embedding)

        with self._connect() as conn:
            next_index = chunk_index if chunk_index is not None else self._next_chunk_index(conn, source)
            cursor = conn.execute(
                """
                INSERT INTO chunks (title, source, content, metadata, embedding, created_at, chunk_index)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    title,
                    source,
                    content,
                    serialized_metadata,
                    serialized_embedding,
                    datetime.utcnow().isoformat(),
                    int(next_index),
                ),
            )
            conn.commit()
            return int(cursor.lastrowid or 0)

    def list_chunks(self, limit: int = 200) -> List[Chunk]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, title, source, content, metadata, embedding, created_at, chunk_index FROM chunks ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()

        chunks: List[Chunk] = []
        for row in rows:
            chunks.append(self._row_to_chunk(row))

        return chunks

    def search(self, query: str, top_k: int = 4, neighbor_window: int = 4) -> List[Chunk]:
        query_embedding = self.embedder(query)
        if not query_embedding:
            return []

        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, title, source, content, metadata, embedding, created_at, chunk_index FROM chunks"
            ).fetchall()

            if not rows:
                return []

            query_vector = np.array(query_embedding)
            query_norm = np.linalg.norm(query_vector)
            if query_norm == 0:
                return []

            scored_chunks: List[Chunk] = []
            for row in rows:
                embedding_payload = row["embedding"]
                if not embedding_payload:
                    continue

                embedding = json.loads(embedding_payload)
                if not embedding:
                    continue

                doc_vector = np.array(embedding)
                doc_norm = np.linalg.norm(doc_vector)
                if doc_norm == 0:
                    continue

                score = float(np.dot(query_vector, doc_vector) / (query_norm * doc_norm))
                scored_chunks.append(self._row_to_chunk(row, score=score))

            scored_chunks.sort(key=lambda chunk: chunk.score or 0.0, reverse=True)
            primary = scored_chunks[:top_k]

            expanded: List[Chunk] = []
            seen: set[int] = set()
            for chunk in primary:
                if chunk.id not in seen:
                    expanded.append(chunk)
                    seen.add(chunk.id)

                neighbors = self._fetch_neighbors(conn, chunk, window=neighbor_window)
                for neighbor in neighbors:
                    if neighbor.id in seen:
                        continue
                    expanded.append(neighbor)
                    seen.add(neighbor.id)

            return expanded

    def _row_to_chunk(self, row: sqlite3.Row, score: Optional[float] = None) -> Chunk:
        metadata = json.loads(row["metadata"]) if row["metadata"] else {}
        chunk_index_value = row["chunk_index"] if "chunk_index" in row.keys() else metadata.get("chunk_index", 0)
        chunk_index = int(chunk_index_value or 0)
        return Chunk(
            id=row["id"],
            title=row["title"],
            source=row["source"],
            content=row["content"],
            metadata=metadata,
            created_at=row["created_at"],
            chunk_index=chunk_index,
            score=score,
        )

    def _fetch_neighbors(
        self,
        conn: sqlite3.Connection,
        chunk: Chunk,
        window: int = 2,
    ) -> List[Chunk]:
        start = max(0, chunk.chunk_index - window)
        end = chunk.chunk_index + window
        rows = conn.execute(
            """
            SELECT id, title, source, content, metadata, embedding, created_at, chunk_index
            FROM chunks
            WHERE source = ? AND chunk_index BETWEEN ? AND ?
            ORDER BY chunk_index
            """,
            (chunk.source, start, end),
        ).fetchall()

        return [self._row_to_chunk(row) for row in rows]

    def count(self) -> int:
        with self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) AS count FROM chunks").fetchone()
        return int(row["count"]) if row else 0

    def _next_chunk_index(self, conn: sqlite3.Connection, source: str) -> int:
        row = conn.execute(
            "SELECT MAX(chunk_index) AS max_idx FROM chunks WHERE source = ?",
            (source,),
        ).fetchone()
        max_idx = row["max_idx"] if row else None
        return int(max_idx + 1) if max_idx is not None else 0

    def reset(self) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM chunks")
            conn.execute("DELETE FROM sqlite_sequence WHERE name='chunks'")
            conn.commit()

    def update_chunk(
        self,
        chunk_id: int,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        embedding = self.embedder(content)
        serialized_embedding = json.dumps(embedding)

        updates = ["content = ?", "embedding = ?"]
        params: List[Any] = [content, serialized_embedding]

        if metadata is not None:
            updates.append("metadata = ?")
            params.append(json.dumps(metadata))

        params.append(chunk_id)

        with self._connect() as conn:
            cursor = conn.execute(
                f"UPDATE chunks SET {', '.join(updates)} WHERE id = ?",
                params
            )
            conn.commit()
            return cursor.rowcount > 0

    def delete_chunk(self, chunk_id: int) -> bool:
        with self._connect() as conn:
            cursor = conn.execute("DELETE FROM chunks WHERE id = ?", (chunk_id,))
            conn.commit()
            return cursor.rowcount > 0

    def get_chunk(self, chunk_id: int) -> Optional[Chunk]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT id, title, source, content, metadata, embedding, created_at, chunk_index FROM chunks WHERE id = ?",
                (chunk_id,),
            ).fetchone()
            if row:
                return self._row_to_chunk(row)
        return None