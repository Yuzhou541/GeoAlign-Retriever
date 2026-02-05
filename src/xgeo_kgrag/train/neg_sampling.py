from __future__ import annotations
import torch

def negative_sample(triples: torch.Tensor, num_entities: int, neg_ratio: int = 1) -> torch.Tensor:
    B = triples.size(0)
    neg = triples.repeat_interleave(neg_ratio, dim=0).clone()
    mask = torch.rand(B * neg_ratio, device=triples.device) < 0.5
    rand_ents = torch.randint(0, num_entities, (B * neg_ratio,), device=triples.device)
    neg[mask, 0] = rand_ents[mask]
    neg[~mask, 2] = rand_ents[~mask]
    return neg
