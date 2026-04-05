"""StockLLM Integration — bridge between Sparks wiki and StockLLM data.

Syncs StockLLM reports, brain insights, and scout picks into a Sparks wiki.
Also exports wiki principles back to StockLLM's brain.db.

Usage:
    from sparks.integrations.stockllm import StockLLMBridge

    bridge = StockLLMBridge(
        stockllm_path="/home/provever0/StockLLM",
        wiki_path="~/.sparks/wiki/stockllm",
    )

    # Ingest recent reports into wiki
    bridge.sync_reports(days=7)

    # Ingest brain insights into wiki
    bridge.sync_insights()

    # Export wiki principles back to brain
    bridge.export_to_brain()

    # Full sync (both directions)
    bridge.full_sync()
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from sparks.wiki import Wiki


class StockLLMBridge:
    """Bidirectional bridge between Sparks Wiki and StockLLM."""

    def __init__(
        self,
        stockllm_path: str | Path = "/home/provever0/StockLLM",
        wiki_path: str | Path = "~/.sparks/wiki/stockllm",
    ):
        self.stockllm = Path(stockllm_path)
        self.wiki = Wiki(wiki_path)

        self.reports_dir = self.stockllm / "reports"
        self.brain_db = self.stockllm / "brain.db"
        self.main_db = self.stockllm / "stockllm.db"

    def sync_reports(self, days: int = 7, report_types: list[str] | None = None) -> dict:
        """Ingest recent StockLLM reports into the wiki.

        Scans reports/ directory for markdown files from the last N days.
        Each report is ingested as raw text with its type as source.
        """
        cutoff = datetime.now() - timedelta(days=days)
        types = report_types or [
            "daily", "daily_kr", "scout", "picks", "weekly",
            "monthly", "ipo", "deep_dive", "earnings",
        ]

        ingested = 0
        for report_type in types:
            type_dir = self.reports_dir / report_type
            if not type_dir.exists():
                continue

            for md_file in sorted(type_dir.glob("*.md")):
                # Check file date
                try:
                    # Try to extract date from filename (YYYY-MM-DD_*)
                    date_str = md_file.stem[:10]
                    file_date = datetime.strptime(date_str, "%Y-%m-%d")
                    if file_date < cutoff:
                        continue
                except ValueError:
                    # Fall back to file mtime
                    if datetime.fromtimestamp(md_file.stat().st_mtime) < cutoff:
                        continue

                text = md_file.read_text()
                if len(text) < 100:
                    continue

                self.wiki.ingest_text(
                    text=text[:15000],  # Limit size
                    source=f"stockllm/{report_type}/{md_file.name}",
                    goal=f"Extract market insights from {report_type} report",
                )
                ingested += 1

        return {"reports_ingested": ingested, "days_scanned": days}

    def sync_insights(self) -> dict:
        """Ingest StockLLM brain insights into the wiki."""
        if not self.brain_db.exists():
            return {"error": "brain.db not found"}

        conn = sqlite3.connect(str(self.brain_db))
        conn.row_factory = sqlite3.Row

        try:
            rows = conn.execute(
                "SELECT domain, content_json, source, confidence, created_at "
                "FROM trading_insights ORDER BY created_at DESC LIMIT 100"
            ).fetchall()
        except sqlite3.OperationalError:
            # Table might not exist or have different name
            try:
                rows = conn.execute(
                    "SELECT domain, content_json, source, confidence, created_at "
                    "FROM insights ORDER BY created_at DESC LIMIT 100"
                ).fetchall()
            except sqlite3.OperationalError:
                conn.close()
                return {"error": "insights table not found"}

        if not rows:
            conn.close()
            return {"insights_ingested": 0}

        # Group by domain
        by_domain: dict[str, list[str]] = {}
        for row in rows:
            domain = row["domain"] or "general"
            content = row["content_json"] or ""
            confidence = row["confidence"] or 0.5
            by_domain.setdefault(domain, []).append(
                f"[{confidence:.0%}] {content[:200]}"
            )

        # Ingest each domain as a batch
        ingested = 0
        for domain, items in by_domain.items():
            text = f"# {domain} Insights\n\n" + "\n".join(f"- {item}" for item in items[:30])
            self.wiki.ingest_text(
                text=text,
                source=f"brain.db/{domain}",
                goal=f"Organize {domain} trading insights",
            )
            ingested += len(items)

        conn.close()
        return {"insights_ingested": ingested, "domains": len(by_domain)}

    def sync_favorites(self) -> dict:
        """Ingest favorites/watchlist status into wiki."""
        if not self.main_db.exists():
            return {"error": "stockllm.db not found"}

        conn = sqlite3.connect(str(self.main_db))
        conn.row_factory = sqlite3.Row

        try:
            rows = conn.execute(
                "SELECT ticker, sector_key, status, score, category, is_leader "
                "FROM favorites WHERE status != '제거' ORDER BY score DESC"
            ).fetchall()
        except sqlite3.OperationalError:
            conn.close()
            return {"error": "favorites table not found"}

        if not rows:
            conn.close()
            return {"favorites_ingested": 0}

        # Format as text
        lines = ["# Current Favorites\n"]
        for row in rows:
            leader = " ★" if row["is_leader"] else ""
            lines.append(
                f"- **{row['ticker']}** [{row['status']}] "
                f"Score: {row['score']:.1f} | {row['category']}{leader} | {row['sector_key']}"
            )

        self.wiki.ingest_text(
            text="\n".join(lines),
            source="stockllm.db/favorites",
            goal="Track current portfolio favorites and their status",
        )

        conn.close()
        return {"favorites_ingested": len(rows)}

    def export_to_brain(self) -> dict:
        """Export wiki principles back to StockLLM brain.db as insights."""
        if not self.brain_db.exists():
            return {"error": "brain.db not found"}

        # Read wiki principles pages
        pages_dir = self.wiki.root / "pages"
        principles = []
        for page in pages_dir.glob("*.md"):
            content = page.read_text()
            if "principle" in content.lower() or "Category: principles" in content:
                principles.append({
                    "title": page.stem,
                    "content": content[:500],
                })

        if not principles:
            return {"exported": 0, "reason": "no principle pages found"}

        conn = sqlite3.connect(str(self.brain_db))
        exported = 0

        for p in principles:
            try:
                conn.execute(
                    "INSERT OR REPLACE INTO trading_insights "
                    "(domain, content_json, source, confidence, created_at) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (
                        "sparks_wiki",
                        json.dumps({"title": p["title"], "content": p["content"]}, ensure_ascii=False),
                        "sparks_wiki",
                        0.7,
                        datetime.now().isoformat(),
                    ),
                )
                exported += 1
            except sqlite3.OperationalError:
                break

        conn.commit()
        conn.close()
        return {"exported": exported}

    def full_sync(self, days: int = 7) -> dict:
        """Full bidirectional sync."""
        results = {}
        results["reports"] = self.sync_reports(days=days)
        results["insights"] = self.sync_insights()
        results["favorites"] = self.sync_favorites()
        results["export"] = self.export_to_brain()
        return results
