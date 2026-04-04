"""Semantic similarity — TF-IDF cosine similarity for principle matching.

No external dependencies. Pure Python.
Used for convergence detection (Round 1 vs Round 2 principles)
and principle deduplication.
"""

from __future__ import annotations

import math
import re
from collections import Counter


# ── Stop words (EN + KR) ──

STOP_WORDS = {
    # English
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "in", "on", "at", "to",
    "for", "of", "and", "or", "but", "not", "with", "by", "from", "as",
    "into", "through", "during", "before", "after", "above", "below",
    "between", "out", "off", "over", "under", "again", "further", "then",
    "once", "here", "there", "when", "where", "why", "how", "all", "each",
    "every", "both", "few", "more", "most", "other", "some", "such", "no",
    "nor", "only", "own", "same", "so", "than", "too", "very", "just",
    "because", "if", "while", "about", "up", "also", "it", "its", "this",
    "that", "these", "those", "i", "we", "you", "he", "she", "they",
    "what", "which", "who", "whom",
    # Korean particles/postpositions
    "이", "그", "저", "의", "가", "을", "를", "은", "는", "에", "에서",
    "으로", "로", "와", "과", "도", "만", "한", "할", "하는", "된", "되는",
    "있다", "없다", "것", "수", "등", "및", "더", "가장", "때", "중",
    "후", "내", "간", "약", "모든", "또한", "이러한", "그러나", "따라서",
}


def _has_korean(text: str) -> bool:
    return bool(re.search(r'[가-힣]', text))


# Korean suffixes to strip for approximate stemming
_KR_SUFFIXES = [
    "합니다", "합니까", "합시다", "합니다",
    "한다", "하다", "된다", "이다",
    "에서", "으로", "에게",
    "는", "은", "을", "를", "이", "가", "의", "에", "도", "만",
    "고", "며", "지", "서", "면",
]


def _kr_stem(word: str) -> str:
    """Approximate Korean stemming by stripping common suffixes."""
    for suffix in _KR_SUFFIXES:
        if word.endswith(suffix) and len(word) > len(suffix) + 1:
            return word[:-len(suffix)]
    return word


def tokenize(text: str) -> list[str]:
    """Split text into meaningful tokens.

    Korean: word stems + character bigrams.
    English: word-level with stop word removal.
    """
    tokens = []
    words = re.findall(r'[\w가-힣]+', text.lower())

    for w in words:
        if w in STOP_WORDS or len(w) <= 1:
            continue
        tokens.append(w)
        # Korean: also add stemmed form
        if _has_korean(w):
            stemmed = _kr_stem(w)
            if stemmed != w and len(stemmed) > 1:
                tokens.append(stemmed)

    # Korean character bigrams for morphological similarity
    if _has_korean(text):
        for word in re.findall(r'[가-힣]{2,}', text):
            for i in range(len(word) - 1):
                tokens.append(word[i:i + 2])

    return tokens


def tfidf_vector(tokens: list[str], idf: dict[str, float]) -> dict[str, float]:
    """Compute TF-IDF vector for a document."""
    tf = Counter(tokens)
    total = len(tokens) if tokens else 1
    return {
        word: (count / total) * idf.get(word, 1.0)
        for word, count in tf.items()
    }


def cosine_similarity(vec_a: dict[str, float], vec_b: dict[str, float]) -> float:
    """Cosine similarity between two sparse vectors."""
    common = set(vec_a) & set(vec_b)
    if not common:
        return 0.0

    dot = sum(vec_a[k] * vec_b[k] for k in common)
    norm_a = math.sqrt(sum(v * v for v in vec_a.values()))
    norm_b = math.sqrt(sum(v * v for v in vec_b.values()))

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot / (norm_a * norm_b)


def compute_idf(documents: list[str]) -> dict[str, float]:
    """Compute IDF scores from a corpus of documents."""
    n_docs = len(documents)
    if n_docs == 0:
        return {}

    doc_freq: Counter = Counter()
    for doc in documents:
        tokens = set(tokenize(doc))
        doc_freq.update(tokens)

    return {
        word: math.log(n_docs / (df + 1)) + 1.0
        for word, df in doc_freq.items()
    }


# ── Public API ──


def semantic_similarity(text_a: str, text_b: str, corpus: list[str] = None) -> float:
    """Compute semantic similarity between two texts.

    Uses TF-IDF cosine similarity. If corpus is provided, IDF is computed
    from the corpus. Otherwise, IDF is computed from the two texts only.
    """
    all_docs = corpus or [text_a, text_b]
    idf = compute_idf(all_docs)

    tokens_a = tokenize(text_a)
    tokens_b = tokenize(text_b)

    vec_a = tfidf_vector(tokens_a, idf)
    vec_b = tfidf_vector(tokens_b, idf)

    return cosine_similarity(vec_a, vec_b)


def find_best_match(query: str, candidates: list[str], threshold: float = 0.3) -> tuple[int, float]:
    """Find the best matching candidate for a query.

    Returns (index, similarity). Returns (-1, 0.0) if no match above threshold.
    """
    if not candidates:
        return -1, 0.0

    corpus = [query] + candidates
    idf = compute_idf(corpus)

    query_tokens = tokenize(query)
    query_vec = tfidf_vector(query_tokens, idf)

    best_idx = -1
    best_sim = 0.0

    for i, candidate in enumerate(candidates):
        cand_tokens = tokenize(candidate)
        cand_vec = tfidf_vector(cand_tokens, idf)
        sim = cosine_similarity(query_vec, cand_vec)
        if sim > best_sim:
            best_sim = sim
            best_idx = i

    if best_sim >= threshold:
        return best_idx, best_sim
    return -1, 0.0


def principle_convergence_llm(
    round1: list[str],
    round2: list[str],
    tracker=None,
) -> tuple[float, list[tuple[str, str, float]]]:
    """LLM-based convergence detection (accurate, handles synonyms).

    One LLM call compares all principles at once. Cost: ~$0.01-0.05.
    """
    if not round1 or not round2:
        return 0.0, []

    try:
        from sparks.llm import llm_call
        r1_text = "\n".join(f"R1-{i+1}: {p}" for i, p in enumerate(round1))
        r2_text = "\n".join(f"R2-{i+1}: {p}" for i, p in enumerate(round2))

        prompt = f"""Compare these two sets of principles. For each principle in Round 1, find the best match in Round 2 (if any).

Round 1:
{r1_text}

Round 2:
{r2_text}

Respond as JSON: {{"matches": [{{"r1": 1, "r2": 2, "similarity": 0.85}}, ...], "convergence": 0.7}}
Only include matches with similarity > 0.3. convergence = overall score 0.0-1.0."""

        result = llm_call(prompt, model="claude-haiku-4-5-20251001", tool="convergence", tracker=tracker)

        import json
        # Extract JSON
        for pattern in [r'\{.*\}']:
            m = re.search(pattern, result, re.DOTALL)
            if m:
                data = json.loads(m.group())
                matches = data.get("matches", [])
                pairs = []
                for match in matches:
                    r1_idx = match.get("r1", 1) - 1
                    r2_idx = match.get("r2", 1) - 1
                    sim = match.get("similarity", 0.5)
                    if 0 <= r1_idx < len(round1) and 0 <= r2_idx < len(round2):
                        pairs.append((round1[r1_idx], round2[r2_idx], sim))
                return data.get("convergence", 0.0), pairs
    except Exception:
        pass

    # Fallback to TF-IDF
    return principle_convergence(round1, round2)


def principle_convergence(round1: list[str], round2: list[str]) -> tuple[float, list[tuple[str, str, float]]]:
    """Measure convergence between two sets of principles.

    Returns:
        (convergence_score, matched_pairs)
        convergence_score: 0.0 (no overlap) to 1.0 (perfect convergence)
        matched_pairs: [(r1_principle, r2_principle, similarity)]
    """
    if not round1 or not round2:
        return 0.0, []

    corpus = round1 + round2
    idf = compute_idf(corpus)

    # Build vectors
    vecs1 = [(p, tfidf_vector(tokenize(p), idf)) for p in round1]
    vecs2 = [(p, tfidf_vector(tokenize(p), idf)) for p in round2]

    # Greedy matching: best pair first
    matched = []
    used2 = set()

    for p1, v1 in vecs1:
        best_sim = 0.0
        best_j = -1
        for j, (p2, v2) in enumerate(vecs2):
            if j in used2:
                continue
            sim = cosine_similarity(v1, v2)
            if sim > best_sim:
                best_sim = sim
                best_j = j

        if best_j >= 0 and best_sim > 0.08:
            matched.append((p1, round2[best_j], best_sim))
            used2.add(best_j)

    # Convergence = avg similarity of matched pairs, weighted by coverage
    if not matched:
        return 0.0, []

    avg_sim = sum(s for _, _, s in matched) / len(matched)
    coverage = len(matched) / max(len(round1), len(round2))
    convergence = avg_sim * coverage

    return convergence, matched
