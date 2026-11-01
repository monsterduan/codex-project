"""Chinese-friendly lexical baseline and optional local semantic RRF retrieval.

The lexical baseline is deliberately not presented as a vector database.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
import math
from pathlib import Path
import re
from typing import Protocol


def tokens(text: str) -> list[str]:
    parts = re.findall(r"[a-z0-9_]+|[\u4e00-\u9fff]+", text.lower())
    result = []
    for part in parts:
        if re.fullmatch(r"[\u4e00-\u9fff]+", part):
            result.extend(part[i:i + 2] for i in range(len(part) - 1))
            if len(part) == 1:
                result.append(part)
        else:
            result.append(part)
    return result


@dataclass(frozen=True)
class Chunk:
    id: str
    source: str
    start_line: int
    end_line: int
    text: str
    sha256: str

    def public(self) -> dict:
        return asdict(self)


def load_corpus(root: Path, max_chars: int = 800) -> list[Chunk]:
    """Split at paragraph boundaries; cite original lines, not generated text."""
    if max_chars < 80:
        raise ValueError("max_chars must be at least 80")
    root = root.resolve()
    if not root.is_dir():
        raise ValueError("corpus directory does not exist")
    result = []
    for path in sorted(root.rglob("*")):
        if path.suffix.lower() not in {".md", ".txt"} or not path.is_file():
            continue
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            continue
        raw = path.read_bytes()
        if len(raw) > 2_000_000:
            raise ValueError(f"document too large: {path.name}")
        lines = raw.decode("utf-8-sig").splitlines()
        source = path.relative_to(root).as_posix()
        digest = hashlib.sha256(raw).hexdigest()
        start = 0
        for i in range(len(lines) + 1):
            boundary = i == len(lines) or not lines[i].strip()
            if boundary and start < i:
                # Long paragraphs split by lines; a single long line is kept intact.
                left = start
                size = 0
                for j in range(start, i):
                    if size and size + len(lines[j]) > max_chars:
                        result.append(Chunk(f"{source}:{left+1}-{j}", source, left+1, j,
                                            "\n".join(lines[left:j]), digest))
                        left, size = j, 0
                    size += len(lines[j])
                result.append(Chunk(f"{source}:{left+1}-{i}", source, left+1, i,
                                    "\n".join(lines[left:i]), digest))
            if boundary:
                start = i + 1
    return result


class DenseRanker(Protocol):
    def rank(self, query: str, chunks: list[Chunk]) -> list[tuple[int, float]]: ...


class LocalSentenceRanker:
    """Explicit, local-files-only embedding adapter. No implicit model download.

    Source: https://sbert.net/docs/package_reference/sentence_transformer/model.html
    """
    def __init__(self, model_path: str):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_path, local_files_only=True,
                                         trust_remote_code=False)
        self._cache_key = None
        self._vectors = None

    def rank(self, query: str, chunks: list[Chunk]) -> list[tuple[int, float]]:
        import numpy as np
        key = tuple((c.id, c.sha256) for c in chunks)
        if key != self._cache_key:
            self._vectors = self.model.encode([c.text for c in chunks], normalize_embeddings=True)
            self._cache_key = key
        q = self.model.encode([query], normalize_embeddings=True)[0]
        scores = np.asarray(self._vectors) @ q
        return sorted(enumerate(scores.tolist()), key=lambda p: (-p[1], p[0]))


class Retriever:
    def __init__(self, chunks: list[Chunk], dense: DenseRanker | None = None):
        self.chunks = chunks
        self.dense = dense
        self.terms = [Counter(tokens(c.text)) for c in chunks]
        self.lengths = [sum(t.values()) for t in self.terms]
        self.avg_length = sum(self.lengths) / max(1, len(chunks)) or 1
        self.df = Counter(t for row in self.terms for t in row)

    def lexical(self, query: str) -> list[tuple[int, float]]:
        q = set(tokens(query))
        scored = []
        for i, row in enumerate(self.terms):
            score = 0.0
            for t in q:
                tf = row[t]
                if not tf:
                    continue
                idf = math.log(1 + (len(self.chunks) - self.df[t] + .5) / (self.df[t] + .5))
                score += idf * tf * 2.5 / (tf + 1.5 * (.25 + .75 * self.lengths[i] / self.avg_length))
            if score > 0:
                scored.append((i, score))
        return sorted(scored, key=lambda p: (-p[1], p[0]))

    def search(self, query: str, k: int = 3) -> list[dict]:
        if not 1 <= k <= 20:
            raise ValueError("k must be between 1 and 20")
        lexical = self.lexical(query)
        lexical_scores = dict(lexical)
        ranks = [lexical]
        if self.dense and self.chunks:
            # Only positive cosine scores enter fusion. This is not a confidence score.
            ranks.append([(i, s) for i, s in self.dense.rank(query, self.chunks) if s > 0])
        fused = Counter()
        for ranked in ranks:
            for rank, (i, _) in enumerate(ranked, 1):
                fused[i] += 1 / (60 + rank)
        winners = sorted(fused, key=lambda i: (-fused[i], i))[:k]
        return [dict(self.chunks[i].public(), score=round(fused[i], 8),
                     lexical_score=round(lexical_scores.get(i, 0), 6)) for i in winners]

    def supported(self, question: str, hits: list[dict], minimum_coverage: float = .2) -> bool:
        """Conservative lexical gate even in hybrid mode; tune on held-out data."""
        q = set(tokens(question))
        if not q or not hits:
            return False
        first = set(tokens(hits[0]["text"]))
        return len(q & first) >= 2 and len(q & first) / len(q) >= minimum_coverage
