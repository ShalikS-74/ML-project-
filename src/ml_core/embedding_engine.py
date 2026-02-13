import hashlib
import logging
import pickle
import re
from pathlib import Path
from typing import List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer

from config import EMBEDDING_MODEL, PROCESSED_DATA_DIR


class EmbeddingEngine:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.model = None
        self.model_name = EMBEDDING_MODEL
        self.cache_dir = PROCESSED_DATA_DIR / "embeddings"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def load_model(self) -> None:
        """Load sentence transformer model."""
        if self.model is not None:
            return
        try:
            self.model = SentenceTransformer(self.model_name)
            self.logger.info(f"Loaded model: {self.model_name}")
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise

    def generate_embedding(self, text: str) -> np.ndarray:
        """Generate a single embedding."""
        batch = self.generate_batch_embeddings([text])
        return batch[0]

    def generate_batch_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for multiple texts efficiently."""
        if not texts:
            return np.array([])

        if self.model is None:
            self.load_model()

        clean_texts = [self.clean_text_for_embedding(text) for text in texts]
        cache_key = self._build_cache_key(clean_texts)
        cached = self.load_cached_embeddings(cache_key)
        if cached is not None and len(cached) == len(clean_texts):
            self.logger.info(f"Loaded cached embeddings for {len(clean_texts)} texts")
            return cached

        embeddings = self.model.encode(
            clean_texts,
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True,
        )
        normalized = self.normalize_embeddings(embeddings)
        self.cache_embeddings(normalized, cache_key)
        return normalized

    def clean_text_for_embedding(self, text: str) -> str:
        """Clean text before embedding generation."""
        text = text or ""
        text = re.sub(r"^\s*\d+[\.\)]\s*", "", text)
        text = re.sub(r"\[\d+\s*marks?\]", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\(\d+\s*marks?\)", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s+", " ", text.strip())
        return text

    def normalize_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """Normalize embeddings for cosine similarity."""
        if embeddings.size == 0:
            return embeddings
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return embeddings / norms

    def cache_embeddings(self, embeddings: np.ndarray, cache_key: str) -> None:
        """Save embeddings to cache."""
        try:
            cache_path = self.cache_dir / f"{cache_key}.pkl"
            with open(cache_path, "wb") as handle:
                payload = {"model": self.model_name, "embeddings": embeddings}
                pickle.dump(payload, handle)
        except Exception as e:
            self.logger.warning(f"Failed to cache embeddings: {e}")

    def load_cached_embeddings(self, cache_key: str) -> Optional[np.ndarray]:
        """Load embeddings from cache if available."""
        cache_path = self.cache_dir / f"{cache_key}.pkl"
        if not cache_path.exists():
            return None

        try:
            with open(cache_path, "rb") as handle:
                payload = pickle.load(handle)
            if payload.get("model") != self.model_name:
                return None
            return payload.get("embeddings")
        except Exception as e:
            self.logger.warning(f"Failed to load cached embeddings: {e}")
            return None

    def _build_cache_key(self, clean_texts: List[str]) -> str:
        """Build stable cache key from model name + question text batch."""
        joined = "\n".join(clean_texts)
        digest = hashlib.sha256(joined.encode("utf-8")).hexdigest()[:16]
        safe_model = self.model_name.replace("/", "_")
        return f"{safe_model}_{digest}"
