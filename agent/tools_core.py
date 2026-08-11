"""
The actual retrieval/business logic, kept separate from the MCP server wiring so it's
independently testable and independently reusable (e.g. directly from Streamlit for the
image-upload path, which doesn't need to go through an LLM tool call).
"""
import json
import random
import sys
from pathlib import Path

import PIL

sys.path.append(str(Path(__file__).parent.parent))
import os

import chromadb
from agent.embeddings import SentenceTransformerEmbedder, TfidfEmbedder, get_image_embedder

BASE_DIR = Path(__file__).parent.parent
CHROMA_PATH = str(BASE_DIR / "chroma_db")

_client = None
_product_embedder = None
_policy_embedder = None
_reviews_by_product = None


def _get_client():
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=CHROMA_PATH)
    return _client


def _make_embedder(vectorizer_filename: str):
    """Same backend switch as agent/embeddings.get_text_embedder(), but each
    TF-IDF collection needs its own persisted vectorizer (fit on that
    collection's vocabulary), so this takes a filename rather than being a
    single shared singleton."""
    if os.getenv("EMBEDDER_BACKEND", "sentence_transformers") == "sentence_transformers":
        print(f"Using text embedder : sentence_transformers")
        return SentenceTransformerEmbedder()
    return TfidfEmbedder(vectorizer_path=f"{CHROMA_PATH}/{vectorizer_filename}")


def _get_product_embedder():
    global _product_embedder
    if _product_embedder is None:
        _product_embedder = _make_embedder("tfidf_vectorizer.pkl")
    return _product_embedder


def _get_policy_embedder():
    global _policy_embedder
    if _policy_embedder is None:
        _policy_embedder = _make_embedder("tfidf_vectorizer_policies.pkl")
    return _policy_embedder


def _get_reviews():
    global _reviews_by_product
    if _reviews_by_product is None:
        reviews = json.loads((BASE_DIR / "data" / "reviews.json").read_text())
        _reviews_by_product = {}
        for r in reviews:
            _reviews_by_product.setdefault(r["product_id"], []).append(r)
    return _reviews_by_product


def search_products(query: str, min_rating: float = 0.0, max_price: float | None = None, top_k: int = 5) -> list[dict]:
    """Search the product catalog by natural-language description, with optional
    minimum rating and maximum price filters."""
    col = _get_client().get_collection("products")
    q_emb = _get_product_embedder().embed_texts([query])
    res = col.query(query_embeddings=q_emb, n_results=min(top_k * 4, 50))

    results = []
    for meta in res["metadatas"][0]:
        if meta["rating"] < min_rating:
            continue
        if max_price is not None and meta["price"] > max_price:
            continue
        results.append(meta)
        if len(results) >= top_k:
            break
    return results


_image_embedder = None


def _get_image_embedder():
    global _image_embedder
    if _image_embedder is None:
        _image_embedder = get_image_embedder()
    return _image_embedder


def search_by_image(image: "PIL.Image.Image", top_k: int = 5) -> list[dict]:
    """Find visually similar products given an uploaded PIL image, via a
    nearest-neighbor lookup against the `product_images` Chroma collection
    (built by data/build_image_index.py).

    Falls back to a helpful error if that collection hasn't been built yet.
    """
    try:
        col = _get_client().get_collection("product_images")
    except Exception:
        return []

    emb = _get_image_embedder().embed_image(image)
    res = col.query(query_embeddings=[emb], n_results=top_k)
    return res["metadatas"][0]


def search_by_image_description(image_description: str, top_k: int = 5) -> list[dict]:
    """Fallback path: find products by a short text description of an uploaded
    image, when no real image embedding index is available. Used by the MCP
    tool as an LLM-friendly wrapper since tool-calling LLMs can't pass raw
    image bytes as arguments — the real pixel-based search happens directly
    from the Streamlit upload handler via search_by_image() above."""
    return search_products(query=image_description, top_k=top_k)


def get_product_reviews_summary(product_id: str) -> str:
    """Retrieve and summarize reviews for a specific product_id."""
    reviews = _get_reviews().get(product_id, [])
    if not reviews:
        return f"No reviews found for product {product_id}."
    avg_stars = sum(r["stars"] for r in reviews) / len(reviews)
    sample = " | ".join(r["text"] for r in reviews[:3])
    return (
        f"{len(reviews)} review(s), average {avg_stars:.1f}/5 stars. "
        f"Sample feedback: {sample}"
    )


def answer_policy_question(query: str, top_k: int = 2) -> str:
    """Retrieve the most relevant policy document chunks for a shipping/returns/
    warranty/payment question."""
    col = _get_client().get_collection("policies")
    q_emb = _get_policy_embedder().embed_texts([query])
    res = col.query(query_embeddings=q_emb, n_results=top_k)
    chunks = []
    for doc, meta in zip(res["documents"][0], res["metadatas"][0]):
        chunks.append(f"[{meta['source']}] {doc}")
    return "\n\n".join(chunks) if chunks else "No relevant policy information found."


_MOCK_STATUSES = ["Processing", "Shipped", "Out for delivery", "Delivered"]


def check_order_status(order_id: str) -> dict:
    """Mocked order lookup for demo purposes — returns a deterministic fake status."""
    idx = sum(ord(c) for c in order_id) % len(_MOCK_STATUSES)
    status = _MOCK_STATUSES[idx]
    if status == "Delivered":
        eta = "Delivered"
    elif status == "Out for delivery":
        eta = "Arriving today"
    else:
        eta = "3-5 business days"

    return {
        "order_id": order_id,
        "status": status,
        "estimated_delivery": eta,
    }