import argparse
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import random
import json
import re


# ----------------------------
# IO helpers
# ----------------------------
def _list_files(raw: Path, limit: int = 120) -> List[str]:
    items = []
    if not raw.exists():
        return items
    for p in raw.rglob("*"):
        if p.is_file():
            items.append(str(p.relative_to(raw)).replace("\\", "/"))
    items.sort()
    return items[:limit]


def _looks_like_count_header(line: str) -> bool:
    s = line.strip()
    return bool(re.fullmatch(r"\d+", s))


def _read_lines(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        return [ln.rstrip("\n") for ln in f]


def _write_text(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_json(path: Path, obj: Dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


# ----------------------------
# Parsing triples (string / 2id)
# ----------------------------
def read_triples_anysep_string(path: Path) -> List[Tuple[str, str, str]]:
    """
    Parse triples in format: h r t (whitespace or tab separated).
    """
    triples: List[Tuple[str, str, str]] = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()  # handles tabs/spaces
            if len(parts) != 3:
                raise ValueError(f"Bad triple line in {path}: {line}")
            h, r, t = parts
            triples.append((h, r, t))
    return triples


def read_id_map(path: Path) -> Dict[int, str]:
    """
    Parse entity2id.txt / relation2id.txt.
    Common formats:
      - first line: count
      - following lines: <name> <id>  OR  <id> <name>
    Returns: id -> name
    """
    lines = _read_lines(path)
    if not lines:
        raise ValueError(f"Empty id map file: {path}")

    idx = 0
    if _looks_like_count_header(lines[0]):
        idx = 1

    id2name: Dict[int, str] = {}
    for ln in lines[idx:]:
        ln = ln.strip()
        if not ln:
            continue
        parts = ln.split()
        if len(parts) < 2:
            continue

        a, b = parts[0], parts[1]
        # decide which is int
        if re.fullmatch(r"-?\d+", a):
            # id name
            _id = int(a)
            name = " ".join(parts[1:])
        elif re.fullmatch(r"-?\d+", b):
            # name id
            _id = int(b)
            name = " ".join(parts[:-1])
        else:
            # can't parse; skip
            continue

        id2name[_id] = name

    if len(id2name) == 0:
        raise ValueError(f"Failed to parse id map: {path}")
    return id2name


def read_triples_2id(path: Path, id2ent: Dict[int, str], id2rel: Dict[int, str], order: str) -> List[Tuple[str, str, str]]:
    """
    Parse OpenKE-style *2id.txt.
    Often:
      first line: count
      then lines: h t r  (OpenKE classic)
    We support order:
      - 'htr' => head tail relation
      - 'hrt' => head relation tail
    Output triples always: (h, r, t) with string names.
    """
    lines = _read_lines(path)
    if not lines:
        raise ValueError(f"Empty triple2id file: {path}")

    idx = 0
    if _looks_like_count_header(lines[0]):
        idx = 1

    triples: List[Tuple[str, str, str]] = []
    for ln in lines[idx:]:
        ln = ln.strip()
        if not ln:
            continue
        parts = ln.split()
        if len(parts) != 3:
            raise ValueError(f"Bad 2id triple line in {path}: {ln}")
        a, b, c = map(int, parts)

        if order == "htr":
            hid, tid, rid = a, b, c
        elif order == "hrt":
            hid, rid, tid = a, b, c
        else:
            raise ValueError(f"Unknown triple2id order: {order}")

        h = id2ent.get(hid, f"ENT_{hid}")
        t = id2ent.get(tid, f"ENT_{tid}")
        r = id2rel.get(rid, f"REL_{rid}")
        triples.append((h, r, t))
    return triples


# ----------------------------
# Dataset detection
# ----------------------------
def _pick_by_priority(cands: List[Path], priority_names: List[str]) -> Optional[Path]:
    """
    Choose a file from candidates by matching filename priority (case-insensitive).
    """
    if not cands:
        return None
    name2path = {p.name.lower(): p for p in cands}
    for nm in priority_names:
        p = name2path.get(nm.lower(), None)
        if p is not None:
            return p
    # fallback: choose shortest relative path, then lexicographic
    cands_sorted = sorted(cands, key=lambda p: (len(str(p)), str(p).lower()))
    return cands_sorted[0]


def detect_splits(raw_dir: Path, train_name: str, valid_name: str, test_name: str) -> Tuple[Path, Path, Path]:
    """
    If explicit names exist anywhere under raw_dir, use them.
    Else auto-detect by common priorities.
    """
    if not raw_dir.exists():
        raise FileNotFoundError(f"raw_dir does not exist: {raw_dir}")

    all_files = [p for p in raw_dir.rglob("*") if p.is_file()]

    # explicit names (if user provided)
    def find_named(nm: str) -> List[Path]:
        nml = nm.lower()
        return [p for p in all_files if p.name.lower() == nml]

    t_exp = find_named(train_name) if train_name else []
    v_exp = find_named(valid_name) if valid_name else []
    te_exp = find_named(test_name) if test_name else []

    if t_exp and v_exp and te_exp:
        # choose closest paths (shortest)
        train_p = sorted(t_exp, key=lambda p: len(str(p)))[0]
        valid_p = sorted(v_exp, key=lambda p: len(str(p)))[0]
        test_p  = sorted(te_exp, key=lambda p: len(str(p)))[0]
        return train_p, valid_p, test_p

    # auto candidates
    train_pri = ["train.txt", "train.tsv", "train"]
    valid_pri = ["valid.txt", "valid.tsv", "valid", "dev.txt", "dev.tsv", "dev"]
    test_pri  = ["test.txt", "test.tsv", "test"]

    # also support *2id*
    train_pri_2id = ["train2id.txt", "train2id.tsv"]
    valid_pri_2id = ["valid2id.txt", "valid2id.tsv", "dev2id.txt", "dev2id.tsv"]
    test_pri_2id  = ["test2id.txt", "test2id.tsv"]

    train_cands = [p for p in all_files if p.name.lower() in set([x.lower() for x in train_pri + train_pri_2id])]
    valid_cands = [p for p in all_files if p.name.lower() in set([x.lower() for x in valid_pri + valid_pri_2id])]
    test_cands  = [p for p in all_files if p.name.lower() in set([x.lower() for x in test_pri + test_pri_2id])]

    train_p = _pick_by_priority(train_cands, train_pri + train_pri_2id)
    valid_p = _pick_by_priority(valid_cands, valid_pri + valid_pri_2id)
    test_p  = _pick_by_priority(test_cands,  test_pri  + test_pri_2id)

    if train_p is None or valid_p is None or test_p is None:
        files_preview = _list_files(raw_dir, limit=120)
        msg = (
            f"Missing split files under {raw_dir}.\n"
            f"Auto-detect failed.\n"
            f"Expected one of:\n"
            f"  train: {train_pri + train_pri_2id}\n"
            f"  valid/dev: {valid_pri + valid_pri_2id}\n"
            f"  test: {test_pri + test_pri_2id}\n"
            f"\nFound files (first 120):\n  - " + "\n  - ".join(files_preview if files_preview else ["<raw_dir empty or not found>"])
        )
        raise FileNotFoundError(msg)

    return train_p, valid_p, test_p


def find_id_maps_near(triple_file: Path) -> Tuple[Optional[Path], Optional[Path]]:
    """
    For *2id* triples, try to locate entity2id / relation2id in the same directory or ancestors.
    """
    cur = triple_file.parent
    for _ in range(4):  # climb up to 4 levels
        e = cur / "entity2id.txt"
        r = cur / "relation2id.txt"
        if e.exists() and r.exists():
            return e, r
        cur = cur.parent
    return None, None


# ----------------------------
# Text generation
# ----------------------------
def normalize_entity_text(e: str) -> str:
    s = e.strip()
    s = s.replace("_", " ")
    if s.startswith("/"):
        s = s.replace("/", " ")
    return s


def normalize_relation_text(r: str) -> str:
    s = r.strip()
    s = s.replace("_", " ")
    if s.startswith("/"):
        s = s.replace("/", " ")
    return s


def collect_entities(*triple_lists: List[Tuple[str, str, str]]) -> List[str]:
    ents = set()
    for triples in triple_lists:
        for h, _, t in triples:
            ents.add(h)
            ents.add(t)
    return sorted(list(ents))


def write_triples_tsv(triples: List[Tuple[str, str, str]], out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="\n") as f:
        for h, r, t in triples:
            f.write(f"{h}\t{r}\t{t}\n")


def write_entity_texts(entities: List[str], out_path: Path):
    with out_path.open("w", encoding="utf-8", newline="\n") as f:
        for e in entities:
            f.write(f"{e}\t{normalize_entity_text(e)}\n")


def make_queries_from_triples(
    triples: List[Tuple[str, str, str]],
    max_queries: int,
    seed: int,
) -> List[Tuple[str, str]]:
    """
    Controlled retrieval:
      q1 = "head [SEP] relation" -> tail
      q2 = "relation [SEP] tail" -> head
    """
    rng = random.Random(seed)
    pool = triples[:]
    rng.shuffle(pool)

    out: List[Tuple[str, str]] = []
    for (h, r, t) in pool:
        q1 = f"{normalize_entity_text(h)} [SEP] {normalize_relation_text(r)}"
        out.append((q1, t))
        q2 = f"{normalize_relation_text(r)} [SEP] {normalize_entity_text(t)}"
        out.append((q2, h))
        if len(out) >= max_queries:
            break
    return out


def write_queries(queries: List[Tuple[str, str]], out_path: Path):
    with out_path.open("w", encoding="utf-8", newline="\n") as f:
        for q, a in queries:
            f.write(f"{q}\t{a}\n")


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", type=str, required=True, help="Folder containing split files (can be nested)")
    ap.add_argument("--out_dir", type=str, required=True, help="Output folder like data/wn18rr")

    ap.add_argument("--train_name", type=str, default="train.txt")
    ap.add_argument("--valid_name", type=str, default="valid.txt")
    ap.add_argument("--test_name", type=str, default="test.txt")

    ap.add_argument("--max_queries", type=int, default=20000, help="queries.tsv size (controlled retrieval)")
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument(
        "--triple2id_order",
        type=str,
        default="htr",
        choices=["htr", "hrt"],
        help="If using *2id.txt: 'htr' means head tail relation (OpenKE common); 'hrt' means head relation tail.",
    )
    ap.add_argument("--dry_run", action="store_true", help="Only detect and print split paths; do not write output")
    args = ap.parse_args()

    raw = Path(args.raw_dir)
    out = Path(args.out_dir)

    train_path, valid_path, test_path = detect_splits(raw, args.train_name, args.valid_name, args.test_name)

    print("[FOUND] train:", str(train_path))
    print("[FOUND] valid:", str(valid_path))
    print("[FOUND] test :", str(test_path))

    if args.dry_run:
        print("[DRY RUN] stop here.")
        return

    def load_split(p: Path) -> List[Tuple[str, str, str]]:
        nm = p.name.lower()
        if "2id" in nm:
            e_map_path, r_map_path = find_id_maps_near(p)
            if e_map_path is None or r_map_path is None:
                raise FileNotFoundError(
                    f"Detected 2id triples ({p}) but cannot find entity2id.txt & relation2id.txt near it.\n"
                    f"Please place them in the same folder or ancestor folder (within 4 levels)."
                )
            id2ent = read_id_map(e_map_path)
            id2rel = read_id_map(r_map_path)
            triples = read_triples_2id(p, id2ent=id2ent, id2rel=id2rel, order=args.triple2id_order)
            return triples
        else:
            return read_triples_anysep_string(p)

    train = load_split(train_path)
    valid = load_split(valid_path)
    test = load_split(test_path)

    out.mkdir(parents=True, exist_ok=True)

    # Write in project format (tab-separated h r t)
    write_triples_tsv(train, out / "train.txt")
    write_triples_tsv(valid, out / "valid.txt")
    write_triples_tsv(test, out / "test.txt")

    # entity texts
    entities = collect_entities(train, valid, test)
    write_entity_texts(entities, out / "entity_texts.tsv")

    # controlled queries from test
    queries = make_queries_from_triples(test, max_queries=args.max_queries, seed=args.seed)
    write_queries(queries, out / "queries.tsv")

    meta = {
        "raw_dir": str(raw),
        "out_dir": str(out),
        "detected_train": str(train_path),
        "detected_valid": str(valid_path),
        "detected_test": str(test_path),
        "num_train": len(train),
        "num_valid": len(valid),
        "num_test": len(test),
        "num_entities": len(entities),
        "num_queries": len(queries),
        "seed": int(args.seed),
        "triple2id_order": args.triple2id_order,
    }
    _write_json(out / "meta.json", meta)
    print("[OK] prepared dataset:")
    print(json.dumps(meta, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
