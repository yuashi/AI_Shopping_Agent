"""
Generates a synthetic product catalog + reviews for local development/testing.

For the real project, replace this with the Amazon Reviews 2023 dataset:
    from datasets import load_dataset
    meta = load_dataset("McAuley-Lab/Amazon-Reviews-2023", "raw_meta_Electronics",
                         split="full", trust_remote_code=True)
    reviews = load_dataset("McAuley-Lab/Amazon-Reviews-2023", "raw_review_Electronics",
                            split="full", trust_remote_code=True)
(Requires internet access to huggingface.co, unavailable in this sandbox.)
"""
import json
import random
from pathlib import Path

random.seed(42)

CATEGORIES = {
    "Electronics": {
        "adjectives": ["Wireless", "Noise-Cancelling", "Portable", "Rechargeable", "Bluetooth", "Smart", "HD", "Compact"],
        "nouns": ["Headphones", "Earbuds", "Speaker", "Webcam", "Keyboard", "Mouse", "Charger", "Power Bank", "Monitor", "SSD Drive"],
        "brands": ["Zenith", "Aurora", "Nexus", "Orbit", "Vertex", "Lumen"],
        "price_range": (15, 300),
    },
    "Home & Kitchen": {
        "adjectives": ["Non-Stick", "Stainless Steel", "Electric", "Adjustable", "Foldable", "Ceramic", "Insulated", "Digital"],
        "nouns": ["Air Fryer", "Blender", "Knife Set", "Coffee Maker", "Cutting Board", "Kettle", "Storage Bin", "Toaster"],
        "brands": ["Hearth&Co", "KitchenPro", "Homely", "CookWell", "Trellis"],
        "price_range": (10, 150),
    },
    "Beauty & Personal Care": {
        "adjectives": ["Hydrating", "Vitamin C", "Fragrance-Free", "SPF 30", "Gentle", "Deep Cleansing", "Anti-Aging", "Organic"],
        "nouns": ["Face Serum", "Moisturizer", "Shampoo", "Sunscreen", "Lip Balm", "Face Wash", "Hair Oil", "Body Lotion"],
        "brands": ["PureGlow", "Botaniq", "Lumière", "Verdant", "Skinly"],
        "price_range": (6, 60),
    },
    "Sports & Outdoors": {
        "adjectives": ["Lightweight", "Breathable", "Waterproof", "Adjustable", "Insulated", "Non-Slip", "Quick-Dry", "Reflective"],
        "nouns": ["Running Shoes", "Yoga Mat", "Water Bottle", "Backpack", "Jacket", "Resistance Bands", "Tent", "Cycling Gloves"],
        "brands": ["TrailBlaze", "PeakForm", "Ridgeline", "Momentum", "Northbound"],
        "price_range": (12, 220),
    },
}

REVIEW_TEMPLATES_POS = [
    "Works exactly as described, {noun_lower} quality feels premium for the price.",
    "Been using this {noun_lower} for a few weeks now and it's held up really well.",
    "Great value. Shipping was fast and the {noun_lower} arrived well packaged.",
    "Exceeded my expectations, would buy again.",
    "Solid build quality, does what it promises.",
]
REVIEW_TEMPLATES_MIXED = [
    "Decent {noun_lower} but the instructions could be clearer.",
    "Good for the price, though I expected slightly better battery life / durability.",
    "It's fine — nothing special but does the job.",
]
REVIEW_TEMPLATES_NEG = [
    "Disappointed, the {noun_lower} stopped working within a week.",
    "Not as described, quality feels cheaper than expected.",
    "Had to return it, didn't fit my needs.",
]


def make_products(n_per_category=40):
    products = []
    pid = 1000
    for category, cfg in CATEGORIES.items():
        for _ in range(n_per_category):
            adj = random.choice(cfg["adjectives"])
            noun = random.choice(cfg["nouns"])
            brand = random.choice(cfg["brands"])
            title = f"{brand} {adj} {noun}"
            price = round(random.uniform(*cfg["price_range"]), 2)
            rating = round(random.uniform(3.0, 5.0), 1)
            review_count = random.randint(5, 2500)
            description = (
                f"The {title} is a {adj.lower()} {noun.lower()} designed for everyday use. "
                f"Features durable construction, easy setup, and a {random.choice(['1-year', '2-year', '90-day'])} warranty. "
                f"Ideal for {random.choice(['home use', 'travel', 'daily commutes', 'gifting', 'office use', 'outdoor activities'])}."
            )
            products.append({
                "product_id": f"P{pid}",
                "title": title,
                "category": category,
                "description": description,
                "price": price,
                "rating": rating,
                "review_count": review_count,
                "image_placeholder": f"https://picsum.photos/seed/{pid}/400/400",
            })
            pid += 1
    return products


def make_reviews(products, max_reviews_per_product=4):
    reviews = []
    rid = 1
    for p in products:
        n = random.randint(1, max_reviews_per_product)
        noun_lower = p["title"].split()[-1].lower()
        for _ in range(n):
            bucket = random.choices(
                [REVIEW_TEMPLATES_POS, REVIEW_TEMPLATES_MIXED, REVIEW_TEMPLATES_NEG],
                weights=[0.6, 0.25, 0.15],
            )[0]
            text = random.choice(bucket).format(noun_lower=noun_lower)
            reviews.append({
                "review_id": f"R{rid}",
                "product_id": p["product_id"],
                "text": text,
                "stars": random.randint(1, 5),
            })
            rid += 1
    return reviews


POLICIES = {
    "shipping_policy.md": """# Shipping Policy

Standard shipping takes 3-5 business days. Express shipping (1-2 business days) is
available at checkout for an additional fee. Orders placed before 2 PM local time
ship the same day. Free standard shipping applies to orders over $35. We currently
only ship within the country; international shipping is not yet supported.
""",
    "returns_policy.md": """# Returns & Refunds Policy

Most items can be returned within 30 days of delivery for a full refund, provided
they are unused and in original packaging. Electronics have a 15-day return window.
Personal care items (opened) are not eligible for return for hygiene reasons.
Refunds are issued to the original payment method within 5-7 business days of us
receiving the returned item.
""",
    "warranty_policy.md": """# Warranty Policy

Electronics come with a manufacturer warranty of 1-2 years as noted on the product
page, covering defects in materials and workmanship. Warranty does not cover
accidental damage, water damage, or normal wear and tear. To file a warranty claim,
contact support with your order ID and a description of the issue.
""",
    "payment_faq.md": """# Payment FAQ

We accept all major credit/debit cards, PayPal, and store gift cards. Payment is
charged when the order ships, not when it's placed. If a payment fails, you'll be
notified by email and given 24 hours to update your payment method before the order
is cancelled. We do not store full card numbers on our servers.
""",
}


def main():
    base = Path(__file__).parent
    products = make_products()
    reviews = make_reviews(products)

    (base / "products.json").write_text(json.dumps(products, indent=2))
    (base / "reviews.json").write_text(json.dumps(reviews, indent=2))

    policies_dir = base / "policies"
    policies_dir.mkdir(exist_ok=True)
    for fname, content in POLICIES.items():
        (policies_dir / fname).write_text(content)

    print(f"Generated {len(products)} products, {len(reviews)} reviews, {len(POLICIES)} policy docs.")


if __name__ == "__main__":
    main()