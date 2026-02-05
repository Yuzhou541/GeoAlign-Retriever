from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import tarfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


# -----------------------------
# Dataset specs (stable sources)
# -----------------------------
# Use ConvE repository tarballs, which contain standard splits for:
#   FB15k-237, WN18RR, YAGO3-10
# Ref: https://github.com/TimDettmers/ConvE  (files include FB15k-237.tar.gz, WN18RR.tar.gz, YAGO3-10.tar.gz)
@dataclass(frozen=True)
class DatasetSpec:
    name: str
    url: str
    archive_name: str
    expected_sha256: Optional[str] = None  # optional; if None, only print computed sha256


SPECS = {
    "wn18rr": DatasetSpec(
        name="wn18rr",
        url="https://github.com/TimDettmers/ConvE/raw/master/WN18RR.tar.gz",
        archive_name="wn18rr.targz",
        expected_sha256=None,
    ),
    "fb15k237": DatasetSpec(
        name="fb15k237",
        url="https://github.com/TimDettmers/ConvE/raw/master/FB15k-237.tar.gz",
        archive_name="fb15k237.targz",
        expected_sha256=None,
    ),
    "yago3-10": DatasetSpec(
        name="yago3-10",
        url="https://github.com/TimDettmers/ConvE/raw/master/YAGO3-10.tar.gz",
        archive_name="yago3-10.targz",
        expected_sha256=None,
    ),
}


# -----------------------------
# Utilities
# -----------------------------
def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _download(url: str, out_path: Path, force: bool = False) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and not force:
        return

    req = Request(
        url,
        headers={
            "User-Agent": "xgeo_kgrag-fetch/1.0 (+https://github.com/)",
            "Accept": "*/*",
        },
        method="GET",
    )

    try:
        with urlopen(req) as r, out_path.open("wb") as f:
            total = r.headers.get("Content-Length")
            total = int(total) if total is not None else None

            downloaded = 0
            last_print = time.time()
            while True:
                chunk = r.read(1024 * 1024)  # 1MB
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)

                now = time.time()
                if total and (now - last_print > 0.2):
                    pct = downloaded * 100.0 / total
                    sys.stdout.write(f"\r[DL] {out_path.name}: {pct:6.2f}%")
                    sys.stdout.flush()
                    last_print = now

            if total:
                sys.stdout.write("\r")
                sys.stdout.flush()

    except HTTPError as e:
        raise RuntimeError(f"Download failed (HTTP {e.code}): {url}") from e
    except URLError as e:
        raise RuntimeError(f"Download failed (URL error): {url}") from e


def _safe_extract_targz(targz_path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(targz_path, "r:gz") as tar:

        def is_within_directory(directory: Path, target: Path) -> bool:
            abs_directory = directory.resolve()
            abs_target = target.resolve()
            return str(abs_target).startswith(str(abs_directory))

        for member in tar.getmembers():
            member_path = out_dir / member.name
            if not is_within_directory(out_dir, member_path):
                raise RuntimeError(f"Unsafe path in tar: {member.name}")

        tar.extractall(path=out_dir)


def _find_split_files(extract_dir: Path) -> Tuple[Path, Path, Path]:
    # Common filenames across sources
    train_candidates = ["train.txt", "train.tsv", "train"]
    valid_candidates = ["valid.txt", "valid.tsv", "valid", "dev.txt", "dev.tsv", "dev"]
    test_candidates = ["test.txt", "test.tsv", "test"]

    def find_one(cands):
        for name in cands:
            hits = list(extract_dir.rglob(name))
            # Prefer shortest path (closer to root)
            if hits:
                hits.sort(key=lambda p: len(p.parts))
                return hits[0]
        return None

    train = find_one(train_candidates)
    valid = find_one(valid_candidates)
    test = find_one(test_candidates)

    if train is None or valid is None or test is None:
        found = [str(p.relative_to(extract_dir)) for p in extract_dir.rglob("*") if p.is_file()]
        msg = (
            f"Could not locate split files under: {extract_dir}\n"
            f"Expected train in {train_candidates}, valid/dev in {valid_candidates}, test in {test_candidates}\n"
            f"Found {len(found)} files (showing up to 80):\n  - "
            + "\n  - ".join(found[:80])
        )
        raise FileNotFoundError(msg)

    return train, valid, test


def _copy_splits(train: Path, valid: Path, test: Path, raw_dir: Path) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(train, raw_dir / "train.txt")
    shutil.copyfile(valid, raw_dir / "valid.txt")
    shutil.copyfile(test, raw_dir / "test.txt")


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, required=True, choices=sorted(SPECS.keys()))
    ap.add_argument("--raw_dir", type=str, required=True)
    ap.add_argument("--cache_dir", type=str, default=".cache/xgeo_kgrag")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    spec = SPECS[args.dataset]
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== Fetch: {spec.name} ===")

    archive_path = cache_dir / spec.archive_name
    extract_dir = cache_dir / f"extract_{spec.name}"

    _download(spec.url, archive_path, force=args.force)
    sha = _sha256(archive_path)
    if spec.expected_sha256 is not None and sha.lower() != spec.expected_sha256.lower():
        raise RuntimeError(
            f"SHA256 mismatch for {archive_path}\n"
            f"expected: {spec.expected_sha256}\n"
            f"got     : {sha}"
        )
    print(f"[OK] downloaded -> {archive_path} (sha256={sha[:12]}...)")

    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    _safe_extract_targz(archive_path, extract_dir)
    print(f"[OK] extracted tar -> {extract_dir}")

    train, valid, test = _find_split_files(extract_dir)
    raw_dir = Path(args.raw_dir)
    _copy_splits(train, valid, test, raw_dir)

    print("[OK] wrote splits:")
    print(f"  - {raw_dir / 'train.txt'}")
    print(f"  - {raw_dir / 'valid.txt'}")
    print(f"  - {raw_dir / 'test.txt'}")
    print("\n[DONE] All requested datasets fetched.")


if __name__ == "__main__":
    main()
