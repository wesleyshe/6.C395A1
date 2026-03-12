"""
MIT Course Catalog — FAISS Indexer
Reads data/courses.json, builds a semantic index, and saves three artifacts:
  data/index.faiss       — FAISS IndexFlatIP (cosine similarity via normalised vectors)
  data/embeddings.npy    — float32 array, shape (N, 384)
  data/courses_meta.json — ordered course list (position i ↔ FAISS id i)

Run once:  python3 retrieval/indexer.py
"""
from __future__ import annotations

import json
import os

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_HERE      = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(_HERE, "..", "data")

COURSES_JSON     = os.path.join(DATA_DIR, "courses.json")
INDEX_PATH       = os.path.join(DATA_DIR, "index.faiss")
EMBEDDINGS_PATH  = os.path.join(DATA_DIR, "embeddings.npy")
META_PATH        = os.path.join(DATA_DIR, "courses_meta.json")

MODEL_NAME  = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE  = 64


# ---------------------------------------------------------------------------
# Text builder
# ---------------------------------------------------------------------------

def build_embed_text(course: dict) -> str:
    """
    Construct the string to embed for a single course.
    Concatenates the most semantically rich fields; omits operational metadata.
    """
    parts = []

    title = course.get("title") or ""
    if title:
        parts.append(title)

    depts = course.get("departments") or []
    if depts:
        parts.append(" ".join(depts))

    level = course.get("level") or []
    if level:
        parts.append(" ".join(level))

    dist = course.get("distribution_requirements") or []
    if dist:
        parts.append(" ".join(dist))

    prereqs = course.get("prerequisites") or ""
    if prereqs and prereqs.strip().lower() not in ("none", ""):
        parts.append(prereqs)

    desc = course.get("description") or ""
    if desc:
        parts.append(desc)

    return " ".join(parts)


# ---------------------------------------------------------------------------
# Build index
# ---------------------------------------------------------------------------

def build_index():
    # ---- Load courses ----
    print(f"Loading courses from {COURSES_JSON} ...")
    with open(COURSES_JSON, encoding="utf-8") as f:
        courses = json.load(f)

    no_desc = sum(1 for c in courses if not c.get("description"))
    print(f"Loaded {len(courses)} courses ({no_desc} without description)")

    # ---- Build embed texts ----
    texts = [build_embed_text(c) for c in courses]

    # ---- Encode ----
    print(f"Loading model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    print(f"Encoding {len(texts)} courses (batch_size={BATCH_SIZE}) ...")
    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,   # unit vectors → IndexFlatIP == cosine similarity
        convert_to_numpy=True,
    )
    embeddings = embeddings.astype("float32")
    print(f"Embeddings shape: {embeddings.shape}, dtype: {embeddings.dtype}")

    # ---- Build FAISS index ----
    dim   = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    print(f"FAISS index built: {index.ntotal} vectors, dim={dim}")

    # ---- Save artifacts ----
    faiss.write_index(index, INDEX_PATH)
    print(f"Saved index     → {INDEX_PATH}  ({os.path.getsize(INDEX_PATH)//1024} KB)")

    np.save(EMBEDDINGS_PATH, embeddings)
    print(f"Saved embeddings → {EMBEDDINGS_PATH}  ({os.path.getsize(EMBEDDINGS_PATH)//1024} KB)")

    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(courses, f, ensure_ascii=False)
    print(f"Saved metadata  → {META_PATH}  ({os.path.getsize(META_PATH)//1024} KB)")

    print("\nDone. Index is ready for search.")


if __name__ == "__main__":
    build_index()
