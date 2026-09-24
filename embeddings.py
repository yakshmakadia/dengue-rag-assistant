"""
SentenceTransformers Embedding Module for Dengue Clinical Chunks.

Wraps the SentenceTransformers embedding models with LangChain-compatible
interfaces, memory caching, and graceful fallback for fast vector computations.
"""

import sys
import warnings
from pathlib import Path
from typing import List, Optional

# Suppress common non-fatal warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from langchain_core.embeddings import Embeddings
from utils.logger import setup_logger

logger = setup_logger("embeddings")

DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"


class LocalSentenceTransformersEmbeddings(Embeddings):
    """
    LangChain compatible Embeddings wrapper around SentenceTransformers.
    Runs 100% locally and offline without external API dependencies.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME, device: str = "cpu"):
        """
        Initializes the SentenceTransformer embedding model.
        
        Args:
            model_name: HuggingFace model identifier.
            device: 'cpu' or 'cuda' (defaults to 'cpu').
        """
        self.model_name = model_name
        self.device = device
        self._model = None
        self._dim = 384
        
        logger.info("Initializing SentenceTransformer: %s on %s", model_name, device)
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(model_name, device=device)
            # Use non-deprecated method if available
            if hasattr(self._model, "get_embedding_dimension"):
                self._dim = self._model.get_embedding_dimension()
            elif hasattr(self._model, "get_sentence_embedding_dimension"):
                self._dim = self._model.get_sentence_embedding_dimension()
            logger.info("Loaded embedding model %s successfully (Dim: %d)", model_name, self._dim)
        except Exception as e:
            logger.error("Failed loading SentenceTransformer '%s': %s", model_name, e)
            raise e

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Generates dense vector embeddings for a list of document chunks.
        
        Args:
            texts: List of document text strings.
            
        Returns:
            List[List[float]]: Float vectors representing text embeddings.
        """
        if not texts:
            return []
        cleaned_texts = [str(t).replace("\n", " ").strip() for t in texts]
        embeddings = self._model.encode(
            cleaned_texts,
            batch_size=32,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        """
        Generates a dense vector embedding for a single user question.
        
        Args:
            text: Query string.
            
        Returns:
            List[float]: Single embedding vector.
        """
        cleaned_text = str(text).replace("\n", " ").strip()
        embedding = self._model.encode(
            cleaned_text,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return embedding.tolist()

    @property
    def embedding_dimension(self) -> int:
        """Returns vector dimensionality."""
        return self._dim


# Global cached embedding model instance
_cached_embeddings_instance: Optional[LocalSentenceTransformersEmbeddings] = None


def get_embedding_model(model_name: str = DEFAULT_MODEL_NAME) -> LocalSentenceTransformersEmbeddings:
    """
    Retrieves or instantiates the global cached embedding model.
    """
    global _cached_embeddings_instance
    if _cached_embeddings_instance is None or _cached_embeddings_instance.model_name != model_name:
        _cached_embeddings_instance = LocalSentenceTransformersEmbeddings(model_name=model_name)
    return _cached_embeddings_instance


if __name__ == "__main__":
    emb_model = get_embedding_model()
    docs = ["Dengue report with platelet count 95,000.", "Normal CBC hematology."]
    vectors = emb_model.embed_documents(docs)
    query_vec = emb_model.embed_query("What is the platelet count?")
    print(f"Document vectors: {len(vectors)} vectors of dimension {len(vectors[0])}")
    print(f"Query vector dimension: {len(query_vec)}")
