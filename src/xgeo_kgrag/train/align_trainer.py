from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, List

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from xgeo_kgrag.data.kg_dataset import KGData
from xgeo_kgrag.data.text_data import load_entity_texts
from xgeo_kgrag.models.text_encoder import TextEncoder, TextEncoderCfg
from xgeo_kgrag.models.projector import MLPProjector
from xgeo_kgrag.models.kge_transe import TransE
from xgeo_kgrag.models.kge_poincare import PoincareKGE
from xgeo_kgrag.utils.device import get_device, get_dtype
from xgeo_kgrag.utils.io import ensure_dir

class PairDataset(Dataset):
    def __init__(self, X: torch.Tensor, Y: torch.Tensor):
        self.X = X
        self.Y = Y
    def __len__(self): return self.X.size(0)
    def __getitem__(self, idx): return self.X[idx], self.Y[idx]

def info_nce(z: torch.Tensor, t: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    z = F.normalize(z, dim=-1)
    t = F.normalize(t, dim=-1)
    logits = z @ t.t() / temperature
    labels = torch.arange(z.size(0), device=z.device)
    return F.cross_entropy(logits, labels)

def _load_kge(cfg: Dict[str, Any], data: KGData, dtype: torch.dtype, device: torch.device):
    ckpt = torch.load(cfg["kge_ckpt"], map_location="cpu")
    kge_cfg = ckpt.get("cfg", {})
    model_name = (kge_cfg.get("kge", {}).get("model", "poincare")).lower()
    dim = int(kge_cfg.get("kge", {}).get("dim", cfg["projector"]["in_dim"]))

    if model_name == "transe":
        model = TransE(data.num_entities, data.num_relations, dim=dim)
        model = model.to(device=device, dtype=dtype)
    else:
        curvature = float(kge_cfg.get("kge", {}).get("curvature", 1.0))
        model = PoincareKGE(data.num_entities, data.num_relations, dim=dim, curvature=curvature, dtype=dtype)
        model = model.to(device=device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()
    return model, model_name, dim

def _save_ckpt(path: Path, projector, optim, epoch: int, best: float, cfg: Dict[str, Any]) -> None:
    ensure_dir(path.parent)
    torch.save({
        "epoch": epoch,
        "best_metric": best,
        "projector": projector.state_dict(),
        "optim": optim.state_dict(),
        "cfg": cfg,
    }, path)

def train_align(cfg: Dict[str, Any]) -> Dict[str, Any]:
    device = get_device(cfg.get("device", "cuda"))
    dtype = get_dtype(cfg.get("dtype", "float32"))

    data = KGData(cfg["data"]["data_dir"])
    et = load_entity_texts(str(Path(cfg["data"]["data_dir"]) / "entity_texts.tsv"))

    ents = data.maps.id2ent
    pairs = [(i, et.ent2text[e]) for i, e in enumerate(ents) if e in et.ent2text]
    if len(pairs) < 4:
        raise RuntimeError("Not enough entity_texts to train aligner. Provide data/<kg>/entity_texts.tsv")

    ent_ids = [i for i,_ in pairs]
    texts = [t for _,t in pairs]

    te_cfg = TextEncoderCfg(**cfg["text_encoder"])
    text_encoder = TextEncoder(te_cfg, device=device)
    with torch.no_grad():
        text_emb = text_encoder.encode(texts, batch_size=int(cfg["data"].get("batch_size", 256))).to(device)

    kge, kge_name, kge_dim = _load_kge(cfg, data, dtype=dtype, device=device)
    struct_all = kge.get_entity_representations_for_alignment().to(device=device, dtype=dtype)
    struct = struct_all[torch.tensor(ent_ids, device=device)]
    assert struct.shape[1] == int(cfg["projector"]["in_dim"]), "projector.in_dim must match kge.dim"

    proj = MLPProjector(**cfg["projector"]).to(device=device, dtype=dtype)
    opt = torch.optim.AdamW(proj.parameters(), lr=float(cfg["train"]["lr"]), weight_decay=float(cfg["train"].get("weight_decay", 0.0)))

    ds = PairDataset(struct, text_emb)
    dl = DataLoader(ds, batch_size=int(cfg["data"]["batch_size"]), shuffle=True, drop_last=True, num_workers=int(cfg["data"].get("num_workers", 0)))

    out_dir = Path(cfg["run"]["out_dir"])
    ckpt_dir = out_dir / "checkpoints"
    ensure_dir(ckpt_dir)

    epochs = int(cfg["train"]["epochs"])
    temp = float(cfg["train"].get("temperature", 0.07))
    log_every = int(cfg["train"].get("log_every", 50))

    best = 1e9
    step = 0
    running = 0.0
    for epoch in range(1, epochs + 1):
        proj.train()
        for X, Y in dl:
            step += 1
            X = X.to(device)
            Y = Y.to(device)
            Z = proj(X)
            loss = info_nce(Z, Y, temperature=temp)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(proj.parameters(), max_norm=5.0)
            opt.step()

            running += loss.item()
            if step % log_every == 0:
                print(f"[align epoch {epoch:03d} step {step:06d}] loss={running/log_every:.4f}")
                running = 0.0

        proj.eval()
        with torch.no_grad():
            Z = proj(struct)
            # --- patched: avoid O(N^2) InfoNCE on huge val sets (e.g., yago3-10 ~123k) ---
            max_val = int(cfg.get('val_max_n', 4096))
            if Z.shape[0] > max_val:
                idx = torch.randperm(Z.shape[0], device=Z.device)[:max_val]
                Z_ = Z.index_select(0, idx)
                T_ = text_emb.index_select(0, idx)
            else:
                Z_, T_ = Z, text_emb
            val_loss = info_nce(Z_, T_, temperature=temp).item()
        print(f"[align eval epoch {epoch:03d}] in_batch_infoNCE={val_loss:.4f}")

        _save_ckpt(ckpt_dir / "last.pt", proj, opt, epoch, best, cfg)
        if val_loss < best:
            best = val_loss
            _save_ckpt(ckpt_dir / "best.pt", proj, opt, epoch, best, cfg)
            print(f"[best] updated best projector: {ckpt_dir/'best.pt'}")

    return {
        "best_infoNCE": best,
        "aligned_entities": len(pairs),
        "kge_model": kge_name,
        "kge_dim": kge_dim,
        "out_dir": str(out_dir),
    }
