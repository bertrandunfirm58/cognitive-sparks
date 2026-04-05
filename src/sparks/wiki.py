"""LLM Wiki — persistent, structured knowledge base that grows with every run.

Based on the LLM Wiki pattern (2026): instead of RAG over raw docs,
the LLM incrementally builds and maintains a wiki of interlinked
markdown pages.

Architecture:
    ┌─────────────┐     ┌──────────────┐     ┌──────────────┐
    │ Raw Sources  │────►│  Sparks 13   │────►│  Wiki Pages  │
    │ (immutable)  │     │  Tools       │     │  (growing)   │
    └─────────────┘     └──────────────┘     └──────┬───────┘
                                                     │
                                              ┌──────▼───────┐
                                              │   index.md   │
                                              │   log.md     │
                                              │   pages/*.md │
                                              └──────────────┘

Three operations:
  - ingest: Process new data → update/create wiki pages
  - query: Search wiki → synthesize answer
  - lint: Health check (contradictions, stale, orphans)

Works standalone or integrated with StockLLM / any external system.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

from sparks.cost import CostTracker, DEPTH_BUDGETS
from sparks.llm import llm_call, llm_structured


# ─── Models ───


class WikiPage(BaseModel):
    """A single wiki page."""
    title: str
    path: str                          # Relative to wiki root
    category: str = "general"          # For index grouping
    content: str = ""
    links: list[str] = []             # Outgoing [[links]]
    sources: list[str] = []           # Raw source references
    created: str = ""
    updated: str = ""
    confidence: float = 0.5


class IngestPlan(BaseModel):
    """LLM-generated plan for updating wiki from new data."""
    pages_to_update: list[dict] = []   # [{title, action, reason, content_delta}]
    pages_to_create: list[dict] = []   # [{title, category, initial_content}]
    log_entry: str = ""


class QueryResult(BaseModel):
    """Result of a wiki query."""
    answer: str
    sources: list[str] = []            # Wiki pages used
    confidence: float = 0.5
    filed_as: str = ""                 # If result was filed as new page


class LintResult(BaseModel):
    """Wiki health check result."""
    contradictions: list[dict] = []    # [{page_a, page_b, issue}]
    stale_pages: list[str] = []        # Pages not updated recently
    orphan_pages: list[str] = []       # Pages with no incoming links
    missing_links: list[dict] = []     # [{from_page, broken_link}]
    suggestions: list[str] = []


# ─── Wiki Engine ───


class Wiki:
    """Persistent, growing knowledge base powered by LLM analysis.

    Usage:
        wiki = Wiki("~/.sparks/wiki/my_project")
        wiki.ingest("./new_data/", goal="Find security patterns")
        wiki.ingest("./more_data/", goal="Update with latest findings")
        answer = wiki.query("What are the top 3 risks?")
        health = wiki.lint()

    StockLLM integration:
        wiki = Wiki("~/.sparks/wiki/stockllm")
        wiki.ingest_text(report_markdown, source="daily_2026-04-05")
        wiki.ingest_text(scout_output, source="scout_AAPL")
    """

    def __init__(self, wiki_path: str | Path):
        self.root = Path(wiki_path).expanduser()
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "pages").mkdir(exist_ok=True)

        self.index_path = self.root / "index.md"
        self.log_path = self.root / "log.md"
        self.schema_path = self.root / "schema.json"

        # Initialize files if needed
        if not self.index_path.exists():
            self.index_path.write_text("# Wiki Index\n\n*Auto-generated. Do not edit manually.*\n\n")
        if not self.log_path.exists():
            self.log_path.write_text("# Wiki Log\n\n")

    # ─── Ingest ───

    def ingest(
        self,
        data_path: str | Path,
        goal: str = "Extract and organize knowledge",
        depth: str = "quick",
        tracker: Optional[CostTracker] = None,
    ) -> dict:
        """Ingest new data using Sparks' 13 tools, then update wiki pages.

        1. Run Sparks on data → get SynthesisOutput
        2. Read current wiki state
        3. LLM generates ingest plan (which pages to update/create)
        4. Apply plan
        5. Update index + log
        """
        if not tracker:
            tracker = CostTracker(DEPTH_BUDGETS[depth])

        # Step 1: Run Sparks analysis
        from sparks.autonomic import run_autonomic
        result = run_autonomic(
            goal=goal,
            data_path=str(data_path),
            depth=depth,
        )

        # Step 2: Generate wiki updates from analysis
        return self._apply_analysis(result, goal, str(data_path), tracker)

    def ingest_text(
        self,
        text: str,
        source: str = "external",
        goal: str = "Extract and organize knowledge",
        tracker: Optional[CostTracker] = None,
    ) -> dict:
        """Ingest raw text directly (no Sparks analysis).

        Useful for:
        - StockLLM reports
        - External documents
        - Manual knowledge entries
        """
        if not tracker:
            tracker = CostTracker(DEPTH_BUDGETS["standard"])

        wiki_state = self._read_wiki_state()

        plan = self._generate_plan(
            new_content=text,
            source=source,
            goal=goal,
            wiki_state=wiki_state,
            tracker=tracker,
        )

        applied = self._apply_plan(plan)
        self._update_index()
        self._append_log(f"Ingested text from {source}: {len(plan.pages_to_update)} updated, "
                         f"{len(plan.pages_to_create)} created")

        return {
            "updated": len(plan.pages_to_update),
            "created": len(plan.pages_to_create),
            "pages": [p.get("title", "") for p in plan.pages_to_update + plan.pages_to_create],
        }

    def _apply_analysis(self, result, goal: str, data_path: str, tracker: CostTracker) -> dict:
        """Convert SynthesisOutput into wiki page updates."""
        from sparks.output import format_output
        analysis_text = format_output(result, goal)

        wiki_state = self._read_wiki_state()

        plan = self._generate_plan(
            new_content=analysis_text,
            source=data_path,
            goal=goal,
            wiki_state=wiki_state,
            tracker=tracker,
        )

        applied = self._apply_plan(plan)
        self._update_index()
        self._append_log(
            f"Sparks analysis ({goal[:50]}): {len(result.principles)} principles → "
            f"{len(plan.pages_to_update)} updated, {len(plan.pages_to_create)} created"
        )

        return {
            "principles": len(result.principles),
            "updated": len(plan.pages_to_update),
            "created": len(plan.pages_to_create),
            "cost": result.total_cost + tracker.total_cost,
        }

    def _generate_plan(
        self,
        new_content: str,
        source: str,
        goal: str,
        wiki_state: str,
        tracker: CostTracker,
    ) -> IngestPlan:
        """LLM generates a plan for updating wiki from new content."""
        prompt = f"""You are a WIKI EDITOR. Given new content and the current wiki state,
plan which pages to update and which to create.

## Current Wiki State
{wiki_state[:8000]}

## New Content to Ingest
{new_content[:12000]}

## Source
{source}

## Goal
{goal}

## Rules
1. UPDATE existing pages when new content adds to or refines existing knowledge
2. CREATE new pages only for genuinely new topics not covered by existing pages
3. Each page should be focused on ONE topic (not a dump of everything)
4. Use [[Page Title]] syntax for cross-references between pages
5. Include source attribution: "Source: {{source}}"
6. For updates: provide the FULL new content for the page (not just the delta)
7. Preserve existing knowledge — ADD to it, don't overwrite unless contradicted
8. Categories: principles, patterns, analogies, entities, methods, observations, contradictions
9. Log entry: one-line summary of what changed

For pages_to_update: {{title, action: "append"|"revise", reason, new_content}}
For pages_to_create: {{title, category, initial_content}}"""

        result = llm_structured(
            prompt,
            model=tracker.select_model("synthesize"),
            schema=IngestPlan,
            tool="wiki_plan",
            tracker=tracker,
            max_tokens=8192,
        )

        return result

    def _apply_plan(self, plan: IngestPlan) -> list[str]:
        """Apply ingest plan to wiki files."""
        applied = []
        now = datetime.now().isoformat()

        for page in plan.pages_to_update:
            title = page.get("title", "")
            slug = _slugify(title)
            path = self.root / "pages" / f"{slug}.md"

            new_content = page.get("new_content", page.get("content_delta", ""))
            if not new_content:
                continue

            if path.exists():
                # Update: replace content but keep frontmatter
                full_content = f"# {title}\n\n{new_content}\n\n---\n*Updated: {now}*\n"
            else:
                full_content = f"# {title}\n\n{new_content}\n\n---\n*Created: {now}*\n"

            path.write_text(full_content)
            applied.append(f"Updated: {title}")

        for page in plan.pages_to_create:
            title = page.get("title", "")
            category = page.get("category", "general")
            content = page.get("initial_content", "")
            slug = _slugify(title)
            path = self.root / "pages" / f"{slug}.md"

            if path.exists():
                continue  # Don't overwrite existing

            full_content = (
                f"# {title}\n\n"
                f"*Category: {category}*\n\n"
                f"{content}\n\n"
                f"---\n*Created: {now}*\n"
            )
            path.write_text(full_content)
            applied.append(f"Created: {title} [{category}]")

        return applied

    # ─── Query ───

    def query(
        self,
        question: str,
        file_result: bool = False,
        tracker: Optional[CostTracker] = None,
    ) -> QueryResult:
        """Search wiki and synthesize an answer.

        Args:
            question: What to ask.
            file_result: If True, save the answer as a new wiki page.
        """
        if not tracker:
            tracker = CostTracker(DEPTH_BUDGETS["standard"])

        # Find relevant pages
        relevant = self._search_pages(question)

        if not relevant:
            return QueryResult(answer="No relevant wiki pages found.", confidence=0.0)

        pages_text = "\n\n".join(
            f"### {p.stem}.md\n{p.read_text()[:3000]}"
            for p in relevant[:10]
        )

        prompt = f"""Answer this question using ONLY the wiki pages below.
Cite your sources as [Page Title].

## Question
{question}

## Wiki Pages
{pages_text}

## Rules
1. Only use information from the wiki pages provided
2. If the wiki doesn't have enough info, say so
3. Cite sources: "According to [Page Title], ..."
4. Rate your confidence (0-1) based on evidence quality"""

        answer = llm_call(
            prompt,
            model=tracker.select_model("synthesize"),
            tool="wiki_query",
            tracker=tracker,
        )

        result = QueryResult(
            answer=answer,
            sources=[p.stem for p in relevant[:10]],
            confidence=0.7,
        )

        if file_result:
            slug = _slugify(f"query_{question[:30]}")
            path = self.root / "pages" / f"{slug}.md"
            path.write_text(
                f"# Query: {question}\n\n{answer}\n\n"
                f"---\n*Filed: {datetime.now().isoformat()}*\n"
            )
            result.filed_as = slug
            self._append_log(f"Query filed: {question[:50]}")
            self._update_index()

        return result

    # ─── Lint ───

    def lint(self, tracker: Optional[CostTracker] = None) -> LintResult:
        """Health check: contradictions, stale pages, orphans, broken links."""
        if not tracker:
            tracker = CostTracker(DEPTH_BUDGETS["quick"])

        pages = list((self.root / "pages").glob("*.md"))
        if not pages:
            return LintResult(suggestions=["Wiki is empty. Run ingest first."])

        # Collect all content + links
        all_content = {}
        all_links = {}
        incoming_links: dict[str, int] = {p.stem: 0 for p in pages}

        for page in pages:
            content = page.read_text()
            all_content[page.stem] = content

            # Extract [[links]]
            links = re.findall(r'\[\[([^\]]+)\]\]', content)
            all_links[page.stem] = links
            for link in links:
                slug = _slugify(link)
                if slug in incoming_links:
                    incoming_links[slug] += 1

        # Orphan pages (no incoming links, not index)
        orphans = [name for name, count in incoming_links.items()
                   if count == 0 and name not in ("index", "log")]

        # Missing links (outgoing link to nonexistent page)
        missing = []
        for page_name, links in all_links.items():
            for link in links:
                slug = _slugify(link)
                if slug not in incoming_links:
                    missing.append({"from_page": page_name, "broken_link": link})

        # Stale pages (not updated in 7+ days)
        stale = []
        for page in pages:
            content = page.read_text()
            updated_match = re.search(r'\*Updated: (\d{4}-\d{2}-\d{2})', content)
            created_match = re.search(r'\*Created: (\d{4}-\d{2}-\d{2})', content)
            date_str = (updated_match or created_match)
            if date_str:
                try:
                    page_date = datetime.fromisoformat(date_str.group(1))
                    if (datetime.now() - page_date).days > 7:
                        stale.append(page.stem)
                except ValueError:
                    pass

        # Contradictions (LLM check if enough pages)
        contradictions = []
        if len(pages) >= 3 and tracker.can_afford(tracker.select_model("synthesize")):
            summary = "\n".join(
                f"[{name}]: {content[:200]}"
                for name, content in list(all_content.items())[:15]
            )
            contra_result = llm_call(
                f"""Check these wiki pages for contradictions.
Return JSON array: [{{"page_a": "...", "page_b": "...", "issue": "..."}}]
If no contradictions, return [].

Pages:
{summary}""",
                model=tracker.select_model("quick_scan"),
                tool="wiki_lint",
                tracker=tracker,
            )
            try:
                parsed = json.loads(contra_result.strip().strip("```json").strip("```"))
                if isinstance(parsed, list):
                    contradictions = parsed
            except (json.JSONDecodeError, ValueError):
                pass

        return LintResult(
            contradictions=contradictions,
            stale_pages=stale,
            orphan_pages=orphans,
            missing_links=missing,
            suggestions=self._generate_suggestions(orphans, stale, missing, contradictions),
        )

    def _generate_suggestions(self, orphans, stale, missing, contradictions) -> list[str]:
        suggestions = []
        if orphans:
            suggestions.append(f"Link to orphan pages: {', '.join(orphans[:5])}")
        if stale:
            suggestions.append(f"Review stale pages: {', '.join(stale[:5])}")
        if missing:
            suggestions.append(f"Create missing pages: {', '.join(set(m['broken_link'] for m in missing[:5]))}")
        if contradictions:
            suggestions.append(f"Resolve {len(contradictions)} contradiction(s)")
        if not suggestions:
            suggestions.append("Wiki is healthy!")
        return suggestions

    # ─── Internal Helpers ───

    def _read_wiki_state(self) -> str:
        """Read current wiki state as context for LLM."""
        parts = []

        # Index
        if self.index_path.exists():
            parts.append(f"## Index\n{self.index_path.read_text()[:2000]}")

        # Page summaries (first 300 chars each)
        pages = sorted((self.root / "pages").glob("*.md"))
        if pages:
            parts.append(f"\n## Pages ({len(pages)} total)")
            for page in pages[:20]:
                content = page.read_text()
                parts.append(f"- **{page.stem}**: {content[:300]}...")

        if not parts:
            parts.append("Wiki is empty. All pages will be new.")

        return "\n".join(parts)

    def _search_pages(self, query: str) -> list[Path]:
        """Simple keyword search across wiki pages."""
        pages = list((self.root / "pages").glob("*.md"))
        if not pages:
            return []

        # Score by keyword overlap
        query_words = set(query.lower().split())
        scored = []
        for page in pages:
            content = page.read_text().lower()
            score = sum(1 for word in query_words if word in content)
            if score > 0:
                scored.append((score, page))

        scored.sort(key=lambda x: -x[0])
        return [p for _, p in scored]

    def _update_index(self):
        """Rebuild index.md from current pages."""
        pages = sorted((self.root / "pages").glob("*.md"))
        categories: dict[str, list[str]] = {}

        for page in pages:
            content = page.read_text()
            # Extract category from content
            cat_match = re.search(r'\*Category: (\w+)\*', content)
            category = cat_match.group(1) if cat_match else "general"
            categories.setdefault(category, []).append(page.stem)

        lines = ["# Wiki Index\n", f"*{len(pages)} pages | Updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*\n"]

        for cat in sorted(categories.keys()):
            lines.append(f"\n## {cat.title()}")
            for name in sorted(categories[cat]):
                lines.append(f"- [[{name}]]")

        self.index_path.write_text("\n".join(lines) + "\n")

    def _append_log(self, entry: str):
        """Append to log.md (chronological, append-only)."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.log_path, "a") as f:
            f.write(f"- `{timestamp}` — {entry}\n")

    # ─── Stats ───

    def stats(self) -> dict:
        """Wiki statistics."""
        pages = list((self.root / "pages").glob("*.md"))
        total_chars = sum(p.read_text().__len__() for p in pages)
        return {
            "pages": len(pages),
            "total_chars": total_chars,
            "categories": len(set(
                re.search(r'\*Category: (\w+)\*', p.read_text()).group(1)
                if re.search(r'\*Category: (\w+)\*', p.read_text())
                else "general"
                for p in pages
            )),
            "wiki_path": str(self.root),
        }

    def list_pages(self) -> list[str]:
        """List all page names."""
        return sorted(p.stem for p in (self.root / "pages").glob("*.md"))


def _slugify(text: str) -> str:
    """Convert title to filename-safe slug."""
    slug = text.lower().strip()
    slug = re.sub(r'[^\w\s-]', '', slug)
    slug = re.sub(r'[\s]+', '_', slug)
    return slug[:80]
