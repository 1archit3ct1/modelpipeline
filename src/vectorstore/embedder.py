"""
embedder.py — Local embedding using nomic-embed-text-v1.5
No API calls. No external services. Weights downloaded once, runs forever.
GPU if available, CPU fallback.
"""

from __future__ import annotations
import os
import time
import logging
from typing import List, Union
import numpy as np

logger = logging.getLogger(__name__)

MODEL_NAME = "nomic-ai/nomic-embed-text-v1.5"
EMBED_DIM = 768
MAX_TOKENS = 8192

_model = None
_tokenizer = None


def _load():
    global _model, _tokenizer
    if _model is not None:
        return
    try:
        from transformers import AutoTokenizer, AutoModel
        import torch

        logger.info(f"Loading embedder: {MODEL_NAME}")
        t0 = time.time()

        _tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            cache_dir=os.environ.get("MODEL_CACHE", "./data/model_cache"),
            trust_remote_code=True,
        )
        _model = AutoModel.from_pretrained(
            MODEL_NAME,
            cache_dir=os.environ.get("MODEL_CACHE", "./data/model_cache"),
            trust_remote_code=True,
        )

        device = "cuda" if torch.cuda.is_available() else "cpu"
        _model = _model.to(device)
        _model.eval()

        logger.info(f"Embedder loaded on {device} in {time.time()-t0:.2f}s")
    except ImportError as e:
        raise RuntimeError(
            f"Missing dependency: {e}\nRun: pip install transformers torch"
        )


def _mean_pool(token_embeddings, attention_mask):
    import torch
    mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)


def embed(texts: Union[str, List[str]], batch_size: int = 64) -> np.ndarray:
    """
    Embed one or more texts. Returns float32 numpy array shape (N, 768).
    Texts are automatically prefixed with 'search_document:' per nomic spec.
    """
    _load()

    import torch
    import torch.nn.functional as F

    if isinstance(texts, str):
        texts = [texts]

    # nomic requires task prefix
    prefixed = [f"search_document: {t}" for t in texts]

    device = next(_model.parameters()).device
    all_embeddings = []

    for i in range(0, len(prefixed), batch_size):
        batch = prefixed[i : i + batch_size]
        encoded = _tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=MAX_TOKENS,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            out = _model(**encoded)

        pooled = _mean_pool(out.last_hidden_state, encoded["attention_mask"])
        normed = F.normalize(pooled, p=2, dim=1)
        all_embeddings.append(normed.cpu().numpy().astype(np.float32))

    return np.vstack(all_embeddings)


def embed_query(text: str) -> np.ndarray:
    """
    Embed a single query string. Uses 'search_query:' prefix per nomic spec.
    Returns shape (768,).
    """
    _load()

    import torch
    import torch.nn.functional as F

    prefixed = f"search_query: {text}"
    device = next(_model.parameters()).device

    encoded = _tokenizer(
        [prefixed],
        padding=True,
        truncation=True,
        max_length=MAX_TOKENS,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        out = _model(**encoded)

    pooled = _mean_pool(out.last_hidden_state, encoded["attention_mask"])
    normed = F.normalize(pooled, p=2, dim=1)
    return normed.cpu().numpy().astype(np.float32)[0]


def dim() -> int:
    return EMBED_DIM
