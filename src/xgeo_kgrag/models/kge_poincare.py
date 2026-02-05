from __future__ import annotations
import torch
import torch.nn as nn

from .kge_base import KGEModel

class PoincareKGE(KGEModel):
    def __init__(self, num_entities: int, num_relations: int, dim: int, curvature: float = 1.0, dtype=torch.float32):
        super().__init__()
        try:
            from geoopt.manifolds.stereographic import PoincareBall
            from geoopt import ManifoldParameter
        except Exception as e:
            raise RuntimeError("geoopt is required for PoincareKGE. Install via `pip install geoopt`.") from e

        self.ball = PoincareBall(c=curvature)
        self.dim = dim
        self.curvature = curvature

        init = torch.randn(num_entities, dim, dtype=dtype) * 1e-3
        init = self.ball.projx(init)
        self.ent = ManifoldParameter(init, manifold=self.ball)

        self.rel_tan = nn.Embedding(num_relations, dim)
        nn.init.uniform_(self.rel_tan.weight, a=-1e-3, b=1e-3)

    def forward(self, triples: torch.Tensor) -> torch.Tensor:
        h = self.ent[triples[:, 0]]
        t = self.ent[triples[:, 2]]
        r_tan = self.rel_tan(triples[:, 1]).to(h.dtype)
        r = self.ball.expmap0(r_tan)
        hr = self.ball.mobius_add(h, r)
        d = self.ball.dist(hr, t)
        return -d

    def get_entity_representations_for_alignment(self) -> torch.Tensor:
        return self.ball.logmap0(self.ent).detach()
