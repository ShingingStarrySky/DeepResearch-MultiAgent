from __future__ import annotations

from typing import List, Optional

import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)


class EmbeddingService:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None

    @property
    def model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(self.model_name)
                logger.info(f"加载嵌入模型: {self.model_name}")
            except ImportError:
                logger.warning(
                    "sentence-transformers 未安装，将使用基于 TF-IDF 的降级方案。"
                    "安装命令: pip install sentence-transformers"
                )
                self._model = None
            except Exception as e:
                logger.error(f"加载嵌入模型失败: {e}")
                self._model = None
        return self._model

    def encode(self, texts: List[str], batch_size: int = 32) -> Optional[np.ndarray]:
        model = self.model
        if model is None:
            return self._fallback_encode(texts)

        try:
            embeddings = model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=len(texts) > 100,
                normalize_embeddings=True,
            )
            return np.array(embeddings)
        except Exception as e:
            logger.error(f"编码失败: {e}")
            return self._fallback_encode(texts)

    def _fallback_encode(self, texts: List[str]) -> Optional[np.ndarray]:
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer

            vectorizer = TfidfVectorizer(max_features=384)
            embeddings = vectorizer.fit_transform(texts).toarray()
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1
            return embeddings / norms
        except Exception as e:
            logger.error(f"降级编码也失败: {e}")
            return None

    def compute_similarity(
        self,
        query_embedding: np.ndarray,
        candidate_embeddings: np.ndarray,
    ) -> np.ndarray:
        return np.dot(candidate_embeddings, query_embedding)

    def compute_similarity_matrix(self, embeddings: np.ndarray) -> np.ndarray:
        return np.dot(embeddings, embeddings.T)
