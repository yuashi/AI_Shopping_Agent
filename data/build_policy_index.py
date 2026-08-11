import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
import chromadb
from agent.embeddings import get_text_embedder


def chunk_text(text, chunk_size=300, overlap=50):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunks.append(" ".join(words[i:i + chunk_size]))
        i += chunk_size - overlap
    return chunks


def main():
    base = Path(__file__).parent
    policies_dir = base / "policies"

    docs, metadatas, ids = [], [], []
    doc_id = 0
    for fpath in sorted(policies_dir.glob("*.md")):
        text = fpath.read_text()
        for chunk in chunk_text(text):
            docs.append(chunk)
            metadatas.append({"source": fpath.name})
            ids.append(f"policy-{doc_id}")
            doc_id += 1

    embedder = get_text_embedder()
    # Reuse the SAME fitted vectorizer as the product index would break this if run
    # standalone with tfidf (different vocab). For the offline demo we fit fresh here;
    # in production (sentence-transformers) this is a non-issue since there's no fitting step.
    if hasattr(embedder, "fit"):
        embedder.vectorizer_path = base.parent / "chroma_db" / "tfidf_vectorizer_policies.pkl"
        embedder._fitted = False
        embedder.fit(docs)
    embeddings = embedder.embed_texts(docs)

    client = chromadb.PersistentClient(path=str(base.parent / "chroma_db"))
    client.delete_collection("policies") if "policies" in [c.name for c in client.list_collections()] else None
    col = client.create_collection("policies")
    col.add(documents=docs, embeddings=embeddings, metadatas=metadatas, ids=ids)

    print(f"Indexed {len(docs)} policy chunks into Chroma collection 'policies'.")


if __name__ == "__main__":
    main()