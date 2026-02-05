from __future__ import annotations
import torch
import torch.nn as nn

from .kge_base import KGEModel

class TransE(KGEModel):
    def __init__(self, num_entities: int, num_relations: int, dim: int, p_norm: int = 1):
        super().__init__()
        self.ent = nn.Embedding(num_entities, dim)
        self.rel = nn.Embedding(num_relations, dim)
        self.p_norm = p_norm
        nn.init.uniform_(self.ent.weight, a=-0.01, b=0.01)
        nn.init.uniform_(self.rel.weight, a=-0.01, b=0.01)

    def forward(self, triples: torch.Tensor) -> torch.Tensor:
        h = self.ent(triples[:, 0])
        r = self.rel(triples[:, 1])
        t = self.ent(triples[:, 2])
        return -torch.linalg.vector_norm(h + r - t, ord=self.p_norm, dim=-1)

    def get_entity_representations_for_alignment(self) -> torch.Tensor:
        return self.ent.weight.detach()
