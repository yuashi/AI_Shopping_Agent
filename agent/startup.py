import sys
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent


def ensure_data_and_indexes():
    products_path = BASE_DIR / "data" / "products.json"
    chroma_path = BASE_DIR / "chroma_db"

    sys.path.append(str(BASE_DIR))

    if not products_path.exists():
        from data.generate_synthetic_data import main as generate_data
        generate_data()

    products_collection_exists = (chroma_path / "chroma.sqlite3").exists()
    if not products_collection_exists:
        from data.build_product_index import main as build_products
        from data.build_policy_index import main as build_policies
        from data.build_image_index import main as build_images
        build_products()
        build_policies()
        build_images()
