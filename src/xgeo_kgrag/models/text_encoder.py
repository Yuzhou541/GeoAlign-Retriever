from __future__ import annotations
from dataclasses import dataclass
from typing import List

import torch

@dataclass
class TextEncoderCfg:
    model_name: str
    max_length: int = 64
    normalize: bool = True

class TextEncoder:
    def __init__(self, cfg: TextEncoderCfg, device: torch.device):
        self.cfg = cfg
        self.device = device
        try:
            from sentence_transformers import SentenceTransformer
            self._st = SentenceTransformer(cfg.model_name, device=str(device))
        except Exception as e:
            raise RuntimeError(
                "Failed to init sentence-transformers. Install it via `pip install sentence-transformers`."
            ) from e

    @torch.no_grad()
    def encode(self, texts: List[str], batch_size: int = 64) -> torch.Tensor:
        emb = self._st.encode(
            texts,
            batch_size=batch_size,
            convert_to_tensor=True,
            show_progress_bar=False,
            normalize_embeddings=self.cfg.normalize,
        )
        return emb.to(self.device)
