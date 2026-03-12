"""
MIT Course Catalog — Hybrid Search
Exposes search_courses(query, filters, top_k) -> list of course dicts.

Hybrid strategy:
  1. Structured pre-filter (department, distribution_requirement, has_prereqs, keyword)
  2. Semantic re-rank of the filtered candidates using cosine similarity

Artifacts required (built by retrieval/indexer.py):
  data/index.faiss
  data/embeddings.npy
  data/courses_meta.json

Usage:
  from retrieval.search import search_courses
  results = search_courses("machine learning", filters={"department": "Course 6"}, top_k=5)
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_HERE     = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.path.join(_HERE, "..", "data")

INDEX_PATH      = os.path.join(DATA_DIR, "index.faiss")
EMBEDDINGS_PATH = os.path.join(DATA_DIR, "embeddings.npy")
META_PATH       = os.path.join(DATA_DIR, "courses_meta.json")

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# ---------------------------------------------------------------------------
# Module-level state (loaded once on first call)
# ---------------------------------------------------------------------------
_index: Optional[faiss.Index] = None
_embeddings: Optional[np.ndarray] = None
_courses: Optional[List[Dict]] = None
_model: Optional[SentenceTransformer] = None
_loaded: bool = False


def _load() -> None:
    """Load all index artifacts and the embedding model into module globals."""
    global _index, _embeddings, _courses, _model, _loaded
    if _loaded:
        return

    for path in (INDEX_PATH, EMBEDDINGS_PATH, META_PATH):
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Index artifact not found: {path}\n"
                "Run  python3 retrieval/indexer.py  first to build the index."
            )

    print("Loading index artifacts ...", flush=True)
    _index      = faiss.read_index(INDEX_PATH)
    _embeddings = np.load(EMBEDDINGS_PATH)          # float32, shape (N, 384)
    with open(META_PATH, encoding="utf-8") as f:
        _courses = json.load(f)

    print(f"Loading model: {MODEL_NAME}", flush=True)
    _model  = SentenceTransformer(MODEL_NAME)
    _loaded = True
    print(f"Ready — {len(_courses)} courses indexed.", flush=True)


# ---------------------------------------------------------------------------
# Structured filtering
# ---------------------------------------------------------------------------

def _has_real_prereqs(course: Dict) -> bool:
    """Return True if the course has actual prerequisites (not None / 'None')."""
    prereq = course.get("prerequisites")
    if prereq is None:
        return False
    return not prereq.strip().lower().startswith("none")


def _apply_filters(
    courses: List[Dict],
    filters: Dict[str, Any],
) -> List[int]:
    """
    Return indices of courses that pass all active filters.
    All filters are ANDed together; a None value means 'no filter on this axis'.

    Supported filter keys:
      department (str)              — case-insensitive substring match on any dept name
      distribution_requirement (str)— exact match against distribution_requirements list
      has_prereqs (bool)            — True: must have prereqs; False: must have none
      keyword (str)                 — case-insensitive substring in course_number OR title
    """
    dept_filter  = (filters.get("department") or "").strip().lower()
    dist_filter  = (filters.get("distribution_requirement") or "").strip()
    prereq_filter = filters.get("has_prereqs")   # True, False, or None
    kw_filter    = (filters.get("keyword") or "").strip().lower()

    result = []
    for i, c in enumerate(courses):
        # ---- department ----
        if dept_filter:
            depts = [d.lower() for d in (c.get("departments") or [])]
            if not any(dept_filter in d for d in depts):
                continue

        # ---- distribution requirement ----
        if dist_filter:
            if dist_filter not in (c.get("distribution_requirements") or []):
                continue

        # ---- has_prereqs ----
        if prereq_filter is not None:
            if prereq_filter != _has_real_prereqs(c):
                continue

        # ---- keyword (course number or title) ----
        if kw_filter:
            num   = (c.get("course_number") or "").lower()
            title = (c.get("title") or "").lower()
            if kw_filter not in num and kw_filter not in title:
                continue

        result.append(i)

    return result


# ---------------------------------------------------------------------------
# Semantic search
# ---------------------------------------------------------------------------

def _semantic_search(
    query_embedding: np.ndarray,
    candidate_ids: List[int],
    top_k: int,
) -> List[int]:
    """
    Return up to top_k indices from candidate_ids ranked by cosine similarity
    to query_embedding.

    Uses FAISS for the unfiltered case and numpy for the filtered case.
    """
    top_k = min(top_k, len(candidate_ids))

    if len(candidate_ids) == len(_courses):
        # Unfiltered — use FAISS directly (most efficient path)
        q = query_embedding.reshape(1, -1)
        _, ids = _index.search(q, top_k)
        return ids[0].tolist()
    else:
        # Filtered — slice embeddings and rank with dot products
        # Use explicit row-wise dot to avoid blas matmul overflow warnings on some platforms
        sub_matrix = _embeddings[candidate_ids].astype("float64")  # (M, 384)
        q64        = query_embedding.astype("float64")
        scores     = np.array([float(np.dot(sub_matrix[i], q64))
                                for i in range(len(sub_matrix))])
        top_local  = np.argsort(scores)[::-1][:top_k]   # descending
        return [candidate_ids[i] for i in top_local]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def search_courses(
    query: str,
    filters: Optional[Dict[str, Any]] = None,
    top_k: int = 10,
) -> List[Dict]:
    """
    Hybrid search: structured pre-filter → semantic re-rank.

    Parameters
    ----------
    query   : Natural-language search query.
    filters : Optional dict with any of:
                department (str)             — e.g. "Electrical Engineering", "Course 6", "math"
                distribution_requirement (str) — e.g. "CI-H", "HASS-H", "REST"
                has_prereqs (bool)           — True/False
                keyword (str)               — substring in course number or title
    top_k   : Maximum number of results to return.

    Returns
    -------
    List of course dicts, ranked by semantic similarity to query.
    """
    _load()

    active_filters = filters or {}
    candidate_ids  = _apply_filters(_courses, active_filters)

    if not candidate_ids:
        return []

    # Encode query with same normalisation as index
    q_emb = _model.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )[0].astype("float32")

    result_ids = _semantic_search(q_emb, candidate_ids, top_k)
    return [_courses[i] for i in result_ids]


# ---------------------------------------------------------------------------
# CLI test harness
# ---------------------------------------------------------------------------

def _print_results(label: str, results: List[Dict]) -> None:
    print(f"\n{'='*60}")
    print(f"Query: {label}")
    print(f"Results: {len(results)}")
    print("=" * 60)
    for i, c in enumerate(results, 1):
        depts    = ", ".join(c.get("departments") or [])
        dist     = ", ".join(c.get("distribution_requirements") or []) or "—"
        prereqs  = c.get("prerequisites") or "None"
        units    = c.get("units") or "—"
        schedule = c.get("schedule") or {}
        sched_str = "; ".join(f"{k}: {v}" for k, v in schedule.items()) if schedule else (
            c.get("schedule_notes") or "—"
        )
        print(f"\n  {i}. [{c.get('course_number')}] {c.get('title')}")
        print(f"     Dept     : {depts}")
        print(f"     Units    : {units}  |  Distrib: {dist}")
        print(f"     Prereqs  : {prereqs[:80]}{'...' if len(prereqs) > 80 else ''}")
        print(f"     Schedule : {sched_str[:100]}{'...' if len(sched_str) > 100 else ''}")
        desc = c.get("description") or "(no description)"
        print(f"     Desc     : {desc[:120]}...")


if __name__ == "__main__":
    # ---- Test query 1: ML courses in Course 6 with no prereqs ----
    results = search_courses(
        query="machine learning",
        filters={"department": "Electrical Engineering", "has_prereqs": False},
        top_k=10,
    )
    _print_results("machine learning courses in Course 6 with no prereqs", results)

    # ---- Test query 2: CI-HW classes about history or philosophy ----
    # Note: the current catalog has CI-HW but no standalone CI-H courses
    results = search_courses(
        query="history philosophy ethics society",
        filters={"distribution_requirement": "CI-HW"},
        top_k=10,
    )
    _print_results("CI-HW classes about history or philosophy", results)

    # ---- Test query 3: introductory biology courses ----
    results = search_courses(
        query="introductory biology",
        filters={"department": "Biology"},
        top_k=10,
    )
    _print_results("introductory biology courses", results)
