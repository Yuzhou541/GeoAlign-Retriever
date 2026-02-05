from __future__ import annotations
import numpy as np
import torch

@torch.no_grad()
def neighborhood_consistency(geo_dists: torch.Tensor, euc_dists: torch.Tensor, k: int = 10) -> float:
    N = geo_dists.size(0)
    geo_knn = torch.topk(geo_dists, k=k+1, largest=False).indices[:, 1:]
    euc_knn = torch.topk(euc_dists, k=k+1, largest=False).indices[:, 1:]

    j = []
    for i in range(N):
        a = set(geo_knn[i].tolist())
        b = set(euc_knn[i].tolist())
        j.append(len(a & b) / max(1, len(a | b)))
    return float(np.mean(j))

@torch.no_grad()
def distance_correlation(geo_d: torch.Tensor, euc_d: torch.Tensor, num_samples: int = 2000) -> float:
    N = geo_d.size(0)
    idx_i = torch.randint(0, N, (num_samples,), device=geo_d.device)
    idx_j = torch.randint(0, N, (num_samples,), device=geo_d.device)
    gd = geo_d[idx_i, idx_j].detach().cpu().numpy()
    ed = euc_d[idx_i, idx_j].detach().cpu().numpy()

    rg = np.argsort(np.argsort(gd))
    re = np.argsort(np.argsort(ed))
    rg = (rg - rg.mean()) / (rg.std() + 1e-9)
    re = (re - re.mean()) / (re.std() + 1e-9)
    return float((rg * re).mean())
