import io
import json
import sys
from pathlib import Path
import requests

sys.path.append(str(Path(__file__).parent.parent))
import chromadb
from PIL import Image, ImageDraw

from agent.embeddings import get_image_embedder


def _get_product_image(product: dict) -> Image.Image:
    resp = requests.get(product["image_placeholder"], timeout=10)
    return Image.open(io.BytesIO(resp.content)).convert("RGB")


def main():
    base = Path(__file__).parent
    products = json.loads((base / "products.json").read_text())

    embedder = get_image_embedder()
    embeddings, metadatas, ids, docs = [], [], [], []

    for p in products:
        img = _get_product_image(p)
        emb = embedder.embed_image(img)
        embeddings.append(emb)
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
        docs.append(p["title"])

    client = chromadb.PersistentClient(path=str(base.parent / "chroma_db"))
    if "product_images" in [c.name for c in client.list_collections()]:
        client.delete_collection("product_images")
    col = client.create_collection("product_images")
    col.add(embeddings=embeddings, metadatas=metadatas, ids=ids, documents=docs)

    print(f"Indexed {len(embeddings)} product images into Chroma collection 'product_images'.")


if __name__ == "__main__":
    main()