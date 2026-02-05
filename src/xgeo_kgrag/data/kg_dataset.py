from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


def _read_triples(path: Path) -> List[Tuple[str, str, str]]:
    """
    Read triples from a file.

    Robustness:
    - Accept both tab-separated and whitespace-separated formats (common across KG releases).
    - Ignore empty lines.
    """
    triples: List[Tuple[str, str, str]] = []
    with path.open("r", encoding="utf-8") as f:
        for ln, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            parts = line.split()  # handles \t and spaces
            if len(parts) != 3:
                raise ValueError(f"Bad triple format at {path} line {ln}: expected 3 columns, got {len(parts)}")
            h, r, t = parts
            triples.append((h, r, t))
    return triples


@dataclass
class KGMaps:
    ent2id: Dict[str, int]
    rel2id: Dict[str, int]
    id2ent: List[str]
    id2rel: List[str]


def build_maps(*triple_lists: List[Tuple[str, str, str]]) -> KGMaps:
    ents = set()
    rels = set()
    for triples in triple_lists:
        for h, r, t in triples:
            ents.add(h)
            ents.add(t)
            rels.add(r)
    id2ent = sorted(list(ents))
    id2rel = sorted(list(rels))
    ent2id = {e: i for i, e in enumerate(id2ent)}
    rel2id = {r: i for i, r in enumerate(id2rel)}
    return KGMaps(ent2id=ent2id, rel2id=rel2id, id2ent=id2ent, id2rel=id2rel)


def encode_triples(triples: List[Tuple[str, str, str]], maps: KGMaps) -> np.ndarray:
    arr = np.zeros((len(triples), 3), dtype=np.int64)
    for i, (h, r, t) in enumerate(triples):
        arr[i, 0] = maps.ent2id[h]
        arr[i, 1] = maps.rel2id[r]
        arr[i, 2] = maps.ent2id[t]
    return arr


class KGETrainDataset(Dataset):
    def __init__(self, triples: np.ndarray):
        self.triples = torch.from_numpy(triples)

    def __len__(self) -> int:
        return int(self.triples.shape[0])

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.triples[idx]


class KGData:
    def __init__(self, data_dir: str):
        d = Path(data_dir)
        self.train_triples = _read_triples(d / "train.txt")
        self.valid_triples = _read_triples(d / "valid.txt")
        self.test_triples = _read_triples(d / "test.txt")
        self.maps = build_maps(self.train_triples, self.valid_triples, self.test_triples)
        self.train = encode_triples(self.train_triples, self.maps)
        self.valid = encode_triples(self.valid_triples, self.maps)
        self.test = encode_triples(self.test_triples, self.maps)

    @property
    def num_entities(self) -> int:
        return len(self.maps.id2ent)

    @property
    def num_relations(self) -> int:
        return len(self.maps.id2rel)
