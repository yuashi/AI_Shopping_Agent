"""
Usage:
    pip install datasets
    python data/build_amazon_dataset.py --category Electronics --n_products 5000
"""
import argparse
import json
from pathlib import Path


CATEGORY_CONFIGS = {
    "Electronics": "raw_meta_Electronics",
    "Beauty": "raw_meta_All_Beauty",
    "Home": "raw_meta_Home_and_Kitchen",
    "Sports": "raw_meta_Sports_and_Outdoors",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", default="Electronics", choices=list(CATEGORY_CONFIGS.keys()))
    parser.add_argument("--n_products", type=int, default=5000)
    parser.add_argument("--max_reviews_per_product", type=int, default=5)
    args = parser.parse_args()

    from datasets import load_dataset

    meta_config = CATEGORY_CONFIGS[args.category]
    review_config = meta_config.replace("raw_meta_", "raw_review_")

    print(f"Loading metadata: {meta_config} ...")
    meta = load_dataset(
        "McAuley-Lab/Amazon-Reviews-2023", meta_config, split="full", trust_remote_code=True
    )
    print(f"Loading reviews: {review_config} ...")
    reviews_ds = load_dataset(
        "McAuley-Lab/Amazon-Reviews-2023", review_config, split="full", trust_remote_code=True
    )

    # Sample products that actually have a title, description, and at least one image
    products = []
    seen_ids = set()
    for row in meta:
        if len(products) >= args.n_products:
            break
        if not row.get("title") or not row.get("parent_asin"):
            continue
        pid = row["parent_asin"]
        if pid in seen_ids:
            continue
        seen_ids.add(pid)

        description = " ".join(row.get("description") or []) or row.get("title", "")
        images = row.get("images") or {}
        image_url = None
        if isinstance(images, dict) and images.get("large"):
            image_url = images["large"][0] if images["large"] else None

        products.append({
            "product_id": pid,
            "title": row["title"],
            "category": args.category,
            "description": description[:1000],
            "price": row.get("price") if isinstance(row.get("price"), (int, float)) else 0.0,
            "rating": row.get("average_rating", 0.0) or 0.0,
            "review_count": row.get("rating_number", 0) or 0,
            "image_placeholder": image_url or f"https://picsum.photos/seed/{pid}/400/400",
        })

    print(f"Kept {len(products)} products with valid titles/ids.")

    # Pull reviews only for the products we kept
    wanted_ids = {p["product_id"] for p in products}
    reviews = []
    counts = {}
    rid = 1
    for row in reviews_ds:
        pid = row.get("parent_asin")
        if pid not in wanted_ids:
            continue
        if counts.get(pid, 0) >= args.max_reviews_per_product:
            continue
        text = row.get("text") or row.get("title") or ""
        if not text:
            continue
        reviews.append({
            "review_id": f"R{rid}",
            "product_id": pid,
            "text": text[:500],
            "stars": row.get("rating", 0) or 0,
        })
        counts[pid] = counts.get(pid, 0) + 1
        rid += 1
        if len(reviews) >= len(products) * args.max_reviews_per_product:
            break

    print(f"Collected {len(reviews)} reviews.")

    base = Path(__file__).parent
    (base / "products.json").write_text(json.dumps(products, indent=2))
    (base / "reviews.json").write_text(json.dumps(reviews, indent=2))
    print(f"Wrote {base / 'products.json'} and {base / 'reviews.json'}.")
    print("Policy docs are hand-written and unaffected — see data/policies/*.md.")


if __name__ == "__main__":
    main()