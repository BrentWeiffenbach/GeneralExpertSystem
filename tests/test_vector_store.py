from pathlib import Path
from typing import List

from ges_core.vector_store import VectorStore


def stub_embed(text: str) -> List[float]:
    score = float(sum(ord(ch) for ch in text))
    return [score, float(len(text) or 1.0)]


def test_vector_store_search(tmp_path: Path) -> None:
    db_path = tmp_path / "chunks.sqlite"
    store = VectorStore(db_path, embedder=stub_embed)

    store.add_chunk(title="Network reset", content="reset router", source="doc")
    store.add_chunk(title="Drive failure", content="replace disk", source="doc")

    results = store.search("router", top_k=1)

    assert len(results) == 2
    assert results[0].title == "Network reset"
    assert sorted(chunk.chunk_index for chunk in results) == [0, 1]
