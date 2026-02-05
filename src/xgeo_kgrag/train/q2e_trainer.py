# src/xgeo_kgrag/train/q2e_trainer.py
from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, List, Tuple
import re
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import faiss

from xgeo_kgrag.data.kg_dataset import KGData
from xgeo_kgrag.eval.retrieval import load_queries
from xgeo_kgrag.models.text_encoder import TextEncoder, TextEncoderCfg
from xgeo_kgrag.models.projector import MLPProjector
from xgeo_kgrag.train.align_trainer import _load_kge  # 复用你已有的加载
from xgeo_kgrag.utils.device import get_device, get_dtype
from xgeo_kgrag.utils.io import ensure_dir


def _norm_ent(s: str) -> str:
    # queries.tsv 里常见空格版实体名，这里转成 ent2id 常用的下划线形式
    return re.sub(r"\s+", "_", s.strip())


def _canon_key(x: str) -> str:
    # canonicalize strings across datasets (yago3-10 / fb15k237 / wn18rr)
    x = x.strip().lower()
    x = re.sub(r"[^0-9a-z]+", " ", x)
    x = re.sub(r"\s+", " ", x).strip()
    return x

def _parse_query(q: str, rel2id, ent2id):
    # NOTE: rel2id/ent2id are *normalized* lookup dicts: canon(key)->id
    parts = [p.strip() for p in q.split("[SEP]")]
    if len(parts) != 2:
        raise ValueError(f"Bad [SEP] split: {q}")
    left, right = parts
    lc = _canon_key(left)
    rc = _canon_key(right)

    # Q_tail: head [SEP] relation -> predict tail
    if rc in rel2id and lc in ent2id:
        return ("tail", ent2id[lc], rel2id[rc])

    # Q_head: relation [SEP] tail -> predict head
    if lc in rel2id and rc in ent2id:
        return ("head", ent2id[rc], rel2id[lc])

    raise ValueError(f"Unrecognized query pattern: {q}")

def _poincare_query_tan(kge, qtype: str, known_eid: int, rid: int) -> torch.Tensor:
    """
    Return q_tan in tangent space at 0, shape (D,)
    Uses proper Möbius translation & inverse.
    """
    ball = kge.ball
    e_ball = kge.ent[known_eid]  # on ball
    r_tan = kge.rel_tan.weight[rid].to(e_ball.dtype)
    r_ball = ball.expmap0(r_tan)
    if qtype == "tail":
        q_ball = ball.mobius_add(e_ball, r_ball)
    else:
        # geoopt compat: some versions lack mobius_neg; Möbius inverse is -x in the ball
        r_inv = -r_ball
        if hasattr(ball, 'projx'):
            r_inv = ball.projx(r_inv)
        q_ball = ball.mobius_add(e_ball, r_inv)
    return ball.logmap0(q_ball)


class QDataset(Dataset):
    def __init__(self, q_texts: List[str], ans_ids: List[int], qinfo: List[Tuple[str,int,int]]):
        self.q_texts = q_texts
        self.ans_ids = ans_ids
        self.qinfo = qinfo
    def __len__(self): return len(self.q_texts)
    def __getitem__(self, i):
        return self.q_texts[i], int(self.ans_ids[i]), self.qinfo[i]


class GeoQueryEncoder(nn.Module):
    def __init__(self, text_dim: int, hidden: int = 512, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * text_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, text_dim),
        )
    def forward(self, q_txt: torch.Tensor, q_geo: torch.Tensor) -> torch.Tensor:
        x = torch.cat([q_txt, q_geo], dim=-1)
        return self.net(x)


def _sampled_infonce(q: torch.Tensor, pos_ids: torch.Tensor, neg_ids: torch.Tensor, ent_vecs: torch.Tensor, tau: float) -> torch.Tensor:
    """
    q: (B,dim)
    pos_ids: (B,)
    neg_ids: (B,M)  # hard negatives
    ent_vecs: (N,dim)
    """
    B, dim = q.shape
    M = neg_ids.size(1)

    pos = ent_vecs.index_select(0, pos_ids)              # (B,dim)
    neg = ent_vecs.index_select(0, neg_ids.reshape(-1)).reshape(B, M, dim)  # (B,M,dim)

    # logits: [pos | negs]
    logits_pos = (q * pos).sum(-1, keepdim=True)         # (B,1)
    logits_neg = torch.einsum("bd,bmd->bm", q, neg)      # (B,M)
    logits = torch.cat([logits_pos, logits_neg], dim=1) / tau
    labels = torch.zeros(B, dtype=torch.long, device=q.device)
    return F.cross_entropy(logits, labels)


def train_q2e(cfg: Dict[str, Any]) -> Dict[str, Any]:
    device = get_device(cfg.get("device", "cuda"))
    dtype  = get_dtype(cfg.get("dtype", "float32"))

    data = KGData(cfg["data"]["data_dir"])
    rel2id = data.maps.rel2id
    ent2id = data.maps.ent2id

    queries = load_queries(str(Path(cfg["data"]["data_dir"]) / "queries.tsv"))

    # build normalized lookup maps (canon -> id)
    rel_norm2id = {_canon_key(k): v for k, v in rel2id.items()}
    ent_norm2id = {_canon_key(k): v for k, v in ent2id.items()}

    q_texts, ans_ids, qinfo = [], [], []
    miss = 0
    for q, ans in queries:
        try:
            info = _parse_query(q, rel_norm2id, ent_norm2id)
        except ValueError:
            miss += 1
            continue

        # answers: try raw key first; fallback to canonical
        if ans in ent2id:
            aid = ent2id[ans]
        else:
            aid = ent_norm2id.get(_canon_key(ans), None)
        if aid is None:
            miss += 1
            continue

        q_texts.append(q)
        ans_ids.append(aid)
        qinfo.append(info)

    print(f"[parse] kept={len(qinfo)}/{len(queries)} miss={miss}")

    te_cfg = TextEncoderCfg(**cfg["text_encoder"])
    text_encoder = TextEncoder(te_cfg, device=device)
    # freeze text encoder (TextEncoder is a wrapper; underlying module is usually _st)
    st = getattr(text_encoder, "_st", None) or getattr(text_encoder, "model", None) or getattr(text_encoder, "encoder", None)
    if st is not None:
        if hasattr(st, "eval"):
            st.eval()
        if hasattr(st, "parameters"):
            for p_ in st.parameters():
                p_.requires_grad_(False)
    
    # load kge + projector(best)
    kge, kge_name, kge_dim = _load_kge(cfg, data, dtype=dtype, device=device)

    proj_ckpt = torch.load(cfg["align_ckpt"], map_location="cpu")
    proj = MLPProjector(**cfg["projector"]).to(device=device, dtype=dtype)
    proj.load_state_dict(proj_ckpt["projector"], strict=True)
    proj.eval()
    for p in proj.parameters():
        p.requires_grad = False

    # entity vectors in text space
    with torch.no_grad():
        struct_all = kge.get_entity_representations_for_alignment().to(device=device, dtype=dtype)
        ent_vecs = proj(struct_all)
        ent_vecs = F.normalize(ent_vecs, dim=-1)
    N, text_dim = ent_vecs.shape

    # build FAISS index for hard negatives (CPU index ok; you也可以后续换 GPU)
    ent_np = ent_vecs.detach().cpu().numpy().astype("float32")
    index = faiss.IndexFlatIP(ent_np.shape[1])
    index.add(ent_np)

    # model to train
    qenc = GeoQueryEncoder(text_dim=text_dim,
                           hidden=int(cfg["train"].get("hidden", 512)),
                           dropout=float(cfg["train"].get("dropout", 0.1))
                          ).to(device=device, dtype=dtype)
    opt = torch.optim.AdamW(qenc.parameters(),
                            lr=float(cfg["train"]["lr"]),
                            weight_decay=float(cfg["train"].get("weight_decay", 0.01)))

    ds = QDataset(q_texts, ans_ids, qinfo)
    dl = DataLoader(ds,
                    batch_size=int(cfg["data"].get("batch_size", 128)),
                    shuffle=True,
                    drop_last=True,
                    num_workers=int(cfg["data"].get("num_workers", 0)))

    out_dir = Path(cfg["run"]["out_dir"])
    ckpt_dir = out_dir / "q2e_checkpoints"
    ensure_dir(ckpt_dir)

    epochs = int(cfg["train"]["epochs"])
    tau = float(cfg["train"].get("temperature", 0.07))
    M = int(cfg["train"].get("hard_topm", 256))
    log_every = int(cfg["train"].get("log_every", 50))

    best = 1e9
    step = 0
    running = 0.0

    for epoch in range(1, epochs + 1):
        qenc.train()
        for batch in dl:
            step += 1
            q_batch, ans_batch, info_batch = batch
            ans_batch = torch.as_tensor(ans_batch, device=device, dtype=torch.long)

            # q_txt
            with torch.no_grad():
                q_txt = text_encoder.encode(list(q_batch), batch_size=len(q_batch)).to(device)
                q_txt = F.normalize(q_txt, dim=-1)

            # q_geo via kge (Poincaré Möbius)
            with torch.no_grad():
                q_tan_list = []
                # info_batch may be collated as (qtypes, known_eids, rids) by DataLoader default collate_fn
                if isinstance(info_batch, (tuple, list)) and len(info_batch) == 3 and isinstance(info_batch[0], (list, tuple)):
                    qtypes, known_eids, rids = info_batch
                    iter_info = zip(qtypes, known_eids, rids)
                else:
                    iter_info = info_batch
                for (qtype, known_eid, rid) in iter_info:
                    if kge_name == "poincare":
                        q_tan = _poincare_query_tan(kge, qtype, int(known_eid), int(rid))
                    else:
                        # TransE fallback: (known +/- rel) in Euclidean; 你后面可改成模型原生
                        E = kge.get_entity_representations_for_alignment().to(device=device, dtype=dtype)
                        Rm = None
                        if hasattr(kge, "rel"):
                            Rm = kge.rel.weight
                        else:
                            # heuristic: shape match
                            for _,p in dict(kge.named_parameters()).items():
                                if p.ndim==2 and p.shape[0]==len(rel2id) and p.shape[1]==E.shape[1]:
                                    Rm=p; break
                        rvec = Rm[int(rid)].to(E.dtype)
                        if qtype == "tail":
                            q_tan = E[int(known_eid)] + rvec
                        else:
                            q_tan = E[int(known_eid)] - rvec
                    q_tan_list.append(q_tan)
                q_tan = torch.stack(q_tan_list, dim=0).to(device=device, dtype=dtype)
                q_geo = proj(q_tan)
                q_geo = F.normalize(q_geo, dim=-1)

            # hard negatives: topM by q_geo against ent_vecs
            with torch.no_grad():
                q_np = q_geo.detach().cpu().numpy().astype("float32")
                _, I = index.search(q_np, M + 8)  # 多取一点，方便过滤正样本
                neg = []
                for i in range(I.shape[0]):
                    cand = [int(x) for x in I[i].tolist() if int(x) != int(ans_batch[i].item())]
                    cand = cand[:M]
                    if len(cand) < M:
                        # 补随机
                        extra = torch.randint(0, N, (M - len(cand),)).tolist()
                        cand += extra
                    neg.append(cand)
                neg_ids = torch.as_tensor(neg, device=device, dtype=torch.long)

            # fused query
            q = qenc(q_txt, q_geo)
            q = F.normalize(q, dim=-1)

            loss = _sampled_infonce(q, ans_batch, neg_ids, ent_vecs, tau=tau)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(qenc.parameters(), 5.0)
            opt.step()

            running += loss.item()
            if step % log_every == 0:
                print(f"[q2e epoch {epoch:03d} step {step:06d}] loss={running/log_every:.4f}")
                running = 0.0

        # save last/best (用 loss 作 proxy；你也可以换成 val recall@10)
        ckpt_last = ckpt_dir / "last.pt"
        torch.save({"epoch": epoch, "qenc": qenc.state_dict(), "cfg": cfg}, ckpt_last)

        if loss.item() < best:
            best = loss.item()
            ckpt_best = ckpt_dir / "best.pt"
            torch.save({"epoch": epoch, "best": best, "qenc": qenc.state_dict(), "cfg": cfg}, ckpt_best)
            print(f"[best] updated: {ckpt_best}")

    return {"best_loss": best, "out_dir": str(out_dir), "kge_model": kge_name, "kge_dim": kge_dim}
