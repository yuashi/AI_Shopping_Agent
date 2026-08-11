import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
import chromadb
from agent.embeddings import get_text_embedder


def main():
    base = Path(__file__).parent
    products = json.loads((base / "products.json").read_text())
    reviews = json.loads((base / "reviews.json").read_text())

    reviews_by_product = {}
    for r in reviews:
        reviews_by_product.setdefault(r["product_id"], []).append(r["text"])

    docs, metadatas, ids = [], [], []
    for p in products:
        review_snippet = " ".join(reviews_by_product.get(p["product_id"], [])[:2])
        doc_text = f"{p['title']}. {p['description']} Category: {p['category']}. {review_snippet}"
        docs.append(doc_text)
        metadatas.append({
            "product_id": p["product_id"],
            "title": p["title"],
            "category": p["category"],
            "price": p["price"],
            "rating": p["rating"],
            "review_count": p["review_count"],
            "image_placeholder": p["image_placeholder"],
        })
        ids.append(p["product_id"])

    embedder = get_text_embedder()
    if hasattr(embedder, "fit"):
        embedder.fit(docs)
    embeddings = embedder.embed_texts(docs)

    client = chromadb.PersistentClient(path=str(base.parent / "chroma_db"))
    client.delete_collection("products") if "products" in [c.name for c in client.list_collections()] else None
    col = client.create_collection("products")
    col.add(documents=docs, embeddings=embeddings, metadatas=metadatas, ids=ids)

    print(f"Indexed {len(docs)} products into Chroma collection 'products'.")


if __name__ == "__main__":
    main()