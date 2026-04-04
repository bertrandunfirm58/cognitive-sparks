"""DataStore — load, chunk, sample, and retrieve raw data."""

from __future__ import annotations

import os
import random
from pathlib import Path


class DataStore:
    """Manages raw data files for observation."""

    def __init__(self, data_path: str):
        self.path = Path(data_path)
        self.items: list[dict] = []
        self._load()

    def _load(self):
        if self.path.is_file():
            self.items = [{"file": str(self.path), "content": self.path.read_text(errors="replace")}]
        elif self.path.is_dir():
            for f in sorted(self.path.iterdir()):
                if f.is_file() and f.suffix in (".txt", ".md", ".json", ".csv", ".py", ".ts", ".js"):
                    try:
                        content = f.read_text(errors="replace")
                        if content.strip():
                            self.items.append({"file": f.name, "content": content})
                    except Exception:
                        continue
        if not self.items:
            raise ValueError(f"No readable data found at {self.path}")

    @property
    def total_items(self) -> int:
        return len(self.items)

    def total_chars(self) -> int:
        return sum(len(item["content"]) for item in self.items)

    def estimated_tokens(self) -> int:
        return self.total_chars() // 4

    def sample(self, ratio: float = 0.1, min_n: int = 3, max_n: int = 20) -> list[dict]:
        n = max(min_n, min(max_n, int(len(self.items) * ratio)))
        n = min(n, len(self.items))
        return random.sample(self.items, n)

    def chunks(self, max_chars: int = 32000) -> list[str]:
        """Split items into chunks that fit token budgets."""
        chunks = []
        current_parts: list[str] = []
        current_size = 0

        for item in self.items:
            header = f"--- File: {item['file']} ---\n"
            text = header + item["content"]
            text_len = len(text)

            if current_size + text_len > max_chars and current_parts:
                chunks.append("\n\n".join(current_parts))
                current_parts = []
                current_size = 0

            # If single item exceeds max, truncate it
            if text_len > max_chars:
                text = text[:max_chars] + "\n... [truncated]"

            current_parts.append(text)
            current_size += len(text)

        if current_parts:
            chunks.append("\n\n".join(current_parts))

        return chunks

    def all_text(self, max_chars: int = 700000) -> str:
        """All data as single string, truncated if needed."""
        parts = []
        total = 0
        for item in self.items:
            header = f"--- File: {item['file']} ---\n"
            text = header + item["content"]
            if total + len(text) > max_chars:
                remaining = max_chars - total
                if remaining > 200:
                    parts.append(text[:remaining] + "\n... [truncated]")
                break
            parts.append(text)
            total += len(text)
        return "\n\n".join(parts)

    def file_list(self) -> str:
        return "\n".join(f"- {item['file']} ({len(item['content'])} chars)" for item in self.items)
