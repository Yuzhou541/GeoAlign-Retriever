from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

@dataclass
class EntityTexts:
    ent2text: Dict[str, str]

def load_entity_texts(path: str) -> EntityTexts:
    p = Path(path)
    ent2text: Dict[str, str] = {}
    if not p.exists():
        return EntityTexts(ent2text={})
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if not line:
                continue
            ent, txt = line.split("\t", 1)
            ent2text[ent] = txt
    return EntityTexts(ent2text=ent2text)

def load_queries(path: str) -> List[Tuple[str, str]]:
    p = Path(path)
    if not p.exists():
        return []
    out: List[Tuple[str,str]] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if not line:
                continue
            q, ans = line.split("\t", 1)
            out.append((q, ans))
    return out
