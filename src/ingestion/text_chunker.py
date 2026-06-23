import re
import logging
from dataclasses import dataclass, field
from typing import Optional

from src.ingestion.document_loader import Document

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    id: str
    document_id: str
    document_name: str
    content: str
    chunk_index: int
    start_char: int
    end_char: int
    metadata: dict = field(default_factory=dict)


class RecursiveTextChunker:
    """Splits text using a hierarchy of separators with overlap."""

    _SEPARATORS = ["\n\n", "\n", ". ", "! ", "? ", ", ", " "]

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64, min_chunk_size: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size

    def _split_by_separator(self, text: str, separator: str) -> list[str]:
        if separator == "":
            return list(text)
        parts = text.split(separator)
        return [p + separator for p in parts[:-1]] + [parts[-1]] if parts else []

    def _merge_splits(self, splits: list[str]) -> list[str]:
        chunks, current, current_len = [], [], 0

        for split in splits:
            split_len = len(split)
            if current_len + split_len > self.chunk_size and current:
                merged = "".join(current).strip()
                if len(merged) >= self.min_chunk_size:
                    chunks.append(merged)
                # Keep overlap
                overlap_text = "".join(current)
                overlap_start = max(0, len(overlap_text) - self.chunk_overlap)
                current = [overlap_text[overlap_start:]]
                current_len = len(current[0])

            current.append(split)
            current_len += split_len

        if current:
            merged = "".join(current).strip()
            if len(merged) >= self.min_chunk_size:
                chunks.append(merged)

        return chunks

    def split_text(self, text: str) -> list[str]:
        final_chunks: list[str] = []
        queue = [text]

        for separator in self._SEPARATORS:
            next_queue = []
            for segment in queue:
                if len(segment) <= self.chunk_size:
                    final_chunks.append(segment)
                else:
                    splits = self._split_by_separator(segment, separator)
                    merged = self._merge_splits(splits)
                    for m in merged:
                        if len(m) <= self.chunk_size:
                            final_chunks.append(m)
                        else:
                            next_queue.append(m)
            queue = next_queue

        final_chunks.extend(queue)
        return [c.strip() for c in final_chunks if c.strip()]

    def chunk_document(self, document: Document) -> list[Chunk]:
        texts = self.split_text(document.content)
        chunks = []
        cursor = 0

        for i, text in enumerate(texts):
            start = document.content.find(text, max(0, cursor - self.chunk_overlap))
            if start == -1:
                start = cursor
            end = start + len(text)

            chunk_id = f"{document.id}-chunk-{i:04d}"
            chunks.append(
                Chunk(
                    id=chunk_id,
                    document_id=document.id,
                    document_name=document.filename,
                    content=text,
                    chunk_index=i,
                    start_char=start,
                    end_char=end,
                    metadata={
                        "chunk_count": len(texts),
                        "file_type": document.file_type,
                    },
                )
            )
            cursor = end

        logger.info(f"'{document.filename}' → {len(chunks)} chunks (size={self.chunk_size}, overlap={self.chunk_overlap})")
        return chunks
