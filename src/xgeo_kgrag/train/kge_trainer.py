from __future__ import annotations
from pathlib import Path
from typing import Dict, Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from xgeo_kgrag.data.kg_dataset import KGData, KGETrainDataset
from xgeo_kgrag.models.kge_transe import TransE
from xgeo_kgrag.models.kge_poincare import PoincareKGE
from xgeo_kgrag.train.neg_sampling import negative_sample
from xgeo_kgrag.utils.device import get_device, get_dtype
from xgeo_kgrag.utils.io import ensure_dir

@torch.no_grad()
def link_prediction_mrr(model, triples: torch.Tensor, num_entities: int, device: torch.device, batch_size: int = 256) -> Dict[str, float]:
    model.eval()
    triples = triples.to(device)
    n = triples.size(0)
    ranks = []
    for i in range(0, n, batch_size):
        b = triples[i:i+batch_size]
        h = b[:,0].unsqueeze(1).repeat(1, num_entities).reshape(-1)
        r = b[:,1].unsqueeze(1).repeat(1, num_entities).reshape(-1)
        t = torch.arange(num_entities, device=device).unsqueeze(0).repeat(b.size(0), 1).reshape(-1)
        cand = torch.stack([h, r, t], dim=1)
        scores = model(cand).view(b.size(0), num_entities)
        true_t = b[:,2]
        true_score = scores.gather(1, true_t.unsqueeze(1))
        rank = (scores > true_score).sum(dim=1) + 1
        ranks.append(rank.detach().cpu())
    ranks = torch.cat(ranks).float()
    mrr = (1.0 / ranks).mean().item()
    h10 = (ranks <= 10).float().mean().item()
    return {"mrr_tail": mrr, "hits10_tail": h10}

def _save_ckpt(path: Path, model, optim, epoch: int, best: float, cfg: Dict[str, Any]) -> None:
    ensure_dir(path.parent)
    torch.save({
        "epoch": epoch,
        "best_metric": best,
        "model": model.state_dict(),
        "optim": optim.state_dict(),
        "cfg": cfg,
    }, path)

def train_kge(cfg: Dict[str, Any]) -> Dict[str, Any]:
    device = get_device(cfg.get("device", "cuda"))
    dtype = get_dtype(cfg.get("dtype", "float32"))

    data = KGData(cfg["data"]["data_dir"])
    ds = KGETrainDataset(data.train)
    dl = DataLoader(
        ds,
        batch_size=int(cfg["data"]["batch_size"]),
        shuffle=True,
        num_workers=int(cfg["data"].get("num_workers", 0)),
        pin_memory = torch.cuda.is_available(),
        drop_last=True
    )

    model_name = cfg["kge"]["model"].lower()
    dim = int(cfg["kge"]["dim"])
    if model_name == "transe":
        model = TransE(data.num_entities, data.num_relations, dim=dim, p_norm=int(cfg["kge"].get("p_norm", 1)))
        model = model.to(device=device, dtype=dtype)
        optim = torch.optim.AdamW(model.parameters(), lr=float(cfg["kge"]["lr"]), weight_decay=float(cfg["kge"].get("weight_decay", 0.0)))
    elif model_name == "poincare":
        model = PoincareKGE(data.num_entities, data.num_relations, dim=dim, curvature=float(cfg["kge"].get("curvature", 1.0)), dtype=dtype)
        model = model.to(device=device)
        try:
            from geoopt.optim import RiemannianAdam
            optim = RiemannianAdam(model.parameters(), lr=float(cfg["kge"]["lr"]), weight_decay=float(cfg["kge"].get("weight_decay", 0.0)))
        except Exception:
            optim = torch.optim.AdamW(model.parameters(), lr=float(cfg["kge"]["lr"]), weight_decay=float(cfg["kge"].get("weight_decay", 0.0)))
    else:
        raise ValueError(f"Unknown kge.model: {model_name}")

    out_dir = Path(cfg["run"]["out_dir"])
    ckpt_dir = out_dir / "checkpoints"
    ensure_dir(ckpt_dir)

    neg_ratio = int(cfg["kge"].get("neg_ratio", 8))
    epochs = int(cfg["kge"].get("epochs", 10))
    log_every = int(cfg["kge"].get("log_every", 100))
    eval_every = int(cfg["kge"].get("eval_every", 1))

    best = -1e9
    best_path = ckpt_dir / "best.pt"
    step = 0
    running = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        for batch in dl:
            step += 1
            batch = batch.to(device)
            neg = negative_sample(batch, data.num_entities, neg_ratio=neg_ratio)

            pos_scores = model(batch)
            neg_scores = model(neg)

            loss = F.softplus(-pos_scores).mean() + F.softplus(neg_scores).mean()

            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optim.step()

            running += loss.item()
            if step % log_every == 0:
                print(f"[epoch {epoch:03d} step {step:06d}] loss={running/log_every:.4f}")
                running = 0.0

        if epoch % eval_every == 0:
            metrics = link_prediction_mrr(model, torch.from_numpy(data.valid), data.num_entities, device=device)
            score = metrics["mrr_tail"]
            print(f"[eval epoch {epoch:03d}] valid_mrr_tail={score:.4f} hits10_tail={metrics['hits10_tail']:.4f}")
            _save_ckpt(ckpt_dir / "last.pt", model, optim, epoch, best, cfg)
            if score > best:
                best = score
                _save_ckpt(best_path, model, optim, epoch, best, cfg)
                print(f"[best] updated best checkpoint: {best_path}")

    return {
        "best_valid_mrr_tail": best,
        "num_entities": data.num_entities,
        "num_relations": data.num_relations,
        "kge_model": model_name,
        "dim": dim,
        "out_dir": str(out_dir),
    }
