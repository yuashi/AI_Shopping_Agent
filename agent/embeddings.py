"""
Embedding backends behind one common interface: `embed_texts(list[str]) -> list[list[float]]`.

- TfidfEmbedder: pure scikit-learn, no downloads, no internet. Used in this sandbox
  and as an offline fallback. NOT what you want in the final deployed project.
- SentenceTransformerEmbedder: the real backend (all-MiniLM-L6-v2). Requires
  internet access to huggingface.co the first time it downloads weights.
- ClipImageEmbedder: real image embedding backend (open_clip ViT-B-32), same idea.

Swap which one gets used in data/build_product_index.py and agent/graph.py by
changing EMBEDDER_BACKEND below (or via the EMBEDDER_BACKEND env var).
"""
import os
import pickle
from pathlib import Path

import numpy as np


class TfidfEmbedder:
    """Offline stand-in. Fit once on the corpus, persist the vectorizer, reuse it
    for query-time embedding so query and corpus vectors live in the same space."""

    def __init__(self, vectorizer_path="chroma_db/tfidf_vectorizer.pkl", dim=384):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.dim = dim
        self.vectorizer_path = Path(vectorizer_path)
        self.vectorizer_path.parent.mkdir(parents=True, exist_ok=True)
        if self.vectorizer_path.exists():
            with open(self.vectorizer_path, "rb") as f:
                self.vectorizer = pickle.load(f)
            self._fitted = True
        else:
            self.vectorizer = TfidfVectorizer(max_features=dim, stop_words="english")
            self._fitted = False

    def fit(self, corpus: list[str]):
        self.vectorizer.fit(corpus)
        self._fitted = True
        with open(self.vectorizer_path, "wb") as f:
            pickle.dump(self.vectorizer, f)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not self._fitted:
            raise RuntimeError("TfidfEmbedder must be fit() on the corpus before embedding.")
        mat = self.vectorizer.transform(texts).toarray()
        # pad/truncate to a fixed dim so it behaves like a dense embedding model
        if mat.shape[1] < self.dim:
            mat = np.pad(mat, ((0, 0), (0, self.dim - mat.shape[1])))
        return mat[:, : self.dim].tolist()


class SentenceTransformerEmbedder:
    """Production text embedder. pip install sentence-transformers first."""

    def __init__(self, model_name="all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts, normalize_embeddings=True).tolist()


class ClipImageEmbedder:
    """Production image embedder. pip install open_clip_torch first."""

    def __init__(self, model_name="ViT-B-32", pretrained="openai"):
        import open_clip
        import torch
        self.torch = torch
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.model.eval()

    def embed_image(self, pil_image) -> list[float]:
        image = self.preprocess(pil_image).unsqueeze(0)
        with self.torch.no_grad():
            feats = self.model.encode_image(image)
            feats /= feats.norm(dim=-1, keepdim=True)
        return feats.squeeze(0).tolist()


class MockImageEmbedder:
    """Offline stand-in for ClipImageEmbedder — no model download, no internet.

    Produces a crude color-histogram + downsampled-pixel feature vector. It can
    only tell images apart by broad color/shape, nowhere near CLIP's semantic
    understanding, but it lets the whole image-search pipeline (index build,
    Chroma storage, nearest-neighbor query, MCP tool, UI) run and be verified
    end-to-end without downloading anything. Swap to ClipImageEmbedder for the
    real deployment — same `embed_image` interface, no other code changes needed.
    """

    def __init__(self, dim=384):
        self.dim = dim

    def embed_image(self, pil_image) -> list[float]:
        img = pil_image.convert("RGB").resize((16, 16))
        arr = np.asarray(img, dtype=np.float32) / 255.0
        pixel_features = arr.flatten()  # 16*16*3 = 768

        hist_r = np.histogram(arr[:, :, 0], bins=32, range=(0, 1))[0]
        hist_g = np.histogram(arr[:, :, 1], bins=32, range=(0, 1))[0]
        hist_b = np.histogram(arr[:, :, 2], bins=32, range=(0, 1))[0]
        hist_features = np.concatenate([hist_r, hist_g, hist_b]).astype(np.float32)
        hist_features /= hist_features.sum() + 1e-8

        features = np.concatenate([pixel_features, hist_features])
        if len(features) < self.dim:
            features = np.pad(features, (0, self.dim - len(features)))
        features = features[: self.dim]
        norm = np.linalg.norm(features) + 1e-8
        return (features / norm).tolist()


def get_text_embedder():
    backend = os.getenv("EMBEDDER_BACKEND", "sentence_transformers")
    print(f"Using text embedder backend: {backend}")
    if backend == "sentence_transformers":
        return SentenceTransformerEmbedder()
    return TfidfEmbedder()


def get_image_embedder():
    backend = os.getenv("EMBEDDER_BACKEND", "sentence_transformers")
    print(f"Using image embedder backend: {backend}")
    if backend == "sentence_transformers":  
        return ClipImageEmbedder()
    return MockImageEmbedder()