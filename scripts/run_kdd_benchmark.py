from __future__ import annotations

import argparse
import copy
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import yaml


IS_WIN = sys.platform.startswith("win")


def load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def dump_yaml(obj: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(obj, sort_keys=False, allow_unicode=True), encoding="utf-8")


def deep_get(d: Dict[str, Any], keys: List[str], default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def deep_set(d: Dict[str, Any], keys: List[str], value):
    cur = d
    for k in keys[:-1]:
        if k not in cur or not isinstance(cur[k], dict):
            cur[k] = {}
        cur = cur[k]
    cur[keys[-1]] = value


def run_cmd(cmd: List[str], dry_run: bool = False) -> None:
    print("\n$ " + " ".join(cmd))
    if dry_run:
        return
    subprocess.check_call(cmd)


def expected_best_ckpt(run_out_dir: Path) -> Path:
    return run_out_dir / "checkpoints" / "best.pt"


def force_safe_dataloader_cfg(cfg: Dict[str, Any]) -> None:
    """
    Avoid CUDA IPC / multiprocessing issues on Windows.
    Safe defaults also ok on Linux, just slower.
    """
    if "data" not in cfg or not isinstance(cfg["data"], dict):
        cfg["data"] = {}

    # Force single-process loading to prevent CUDA tensors crossing processes.
    cfg["data"]["num_workers"] = 0

    # Also disable pinned/persistent to reduce IPC + background worker behavior.
    cfg["data"]["pin_memory"] = False
    cfg["data"]["persistent_workers"] = False

    # Some codebases read these keys; harmless if unused.
    cfg["data"]["prefetch_factor"] = 2
    cfg["data"]["drop_last"] = False


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--datasets", nargs="+", default=["wn18rr", "fb15k237", "yago3-10"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    ap.add_argument("--models", nargs="+", default=["poincare", "transe"])
    ap.add_argument("--dims", nargs="+", type=int, default=[32, 64])

    ap.add_argument("--base_kge", type=str, default="configs/kge_poincare.yaml")
    ap.add_argument("--base_align", type=str, default="configs/align_toy.yaml")
    ap.add_argument("--base_eval", type=str, default="configs/eval_toy.yaml")

    ap.add_argument("--out_root", type=str, default="runs/kdd_bench")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--dtype", type=str, default="float32")

    ap.add_argument("--skip_kge", action="store_true")
    ap.add_argument("--skip_align", action="store_true")
    ap.add_argument("--skip_eval", action="store_true")

    ap.add_argument("--force", action="store_true", help="Force rerun even if outputs exist")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    root = Path(args.out_root)
    root.mkdir(parents=True, exist_ok=True)

    base_kge = load_yaml(Path(args.base_kge))
    base_align = load_yaml(Path(args.base_align))
    base_eval = load_yaml(Path(args.base_eval))

    py = sys.executable

    for ds in args.datasets:
        data_dir = Path("data") / ds
        if not data_dir.exists():
            raise FileNotFoundError(f"Missing dataset dir: {data_dir} (expect prepared data/{ds})")

        for seed in args.seeds:
            for model in args.models:
                for dim in args.dims:
                    tag = f"{ds}/{model}_d{dim}_s{seed}"
                    run_dir = root / ds / f"{model}_d{dim}_s{seed}"
                    cfg_dir = run_dir / "configs"
                    cfg_dir.mkdir(parents=True, exist_ok=True)

                    # -------------------
                    # 1) KGE config
                    # -------------------
                    kge_cfg = copy.deepcopy(base_kge)
                    deep_set(kge_cfg, ["seed"], seed)
                    deep_set(kge_cfg, ["device"], args.device)
                    deep_set(kge_cfg, ["dtype"], args.dtype)
                    deep_set(kge_cfg, ["data", "data_dir"], str(data_dir))
                    deep_set(kge_cfg, ["run", "out_dir"], str(run_dir / "kge"))
                    deep_set(kge_cfg, ["kge", "model"], model)
                    deep_set(kge_cfg, ["kge", "dim"], int(dim))
                    if model == "poincare":
                        if deep_get(kge_cfg, ["kge", "curvature"], None) is None:
                            deep_set(kge_cfg, ["kge", "curvature"], 1.0)

                    force_safe_dataloader_cfg(kge_cfg)

                    kge_cfg_path = cfg_dir / "kge.yaml"
                    dump_yaml(kge_cfg, kge_cfg_path)

                    kge_out = Path(deep_get(kge_cfg, ["run", "out_dir"]))
                    kge_ckpt = expected_best_ckpt(kge_out)

                    # -------------------
                    # 2) Align config
                    # -------------------
                    align_cfg = copy.deepcopy(base_align)
                    deep_set(align_cfg, ["seed"], seed)
                    deep_set(align_cfg, ["device"], args.device)
                    deep_set(align_cfg, ["dtype"], args.dtype)
                    deep_set(align_cfg, ["data", "data_dir"], str(data_dir))
                    deep_set(align_cfg, ["run", "out_dir"], str(run_dir / "align"))
                    deep_set(align_cfg, ["kge_ckpt"], str(kge_ckpt))

                    # *** CRITICAL FIX ***
                    # projector.in_dim must match current kge dim
                    if "projector" not in align_cfg or not isinstance(align_cfg["projector"], dict):
                        raise RuntimeError(
                            f"base_align ({args.base_align}) missing root-level 'projector' dict, "
                            f"but align_trainer expects cfg['projector']['in_dim']."
                        )
                    align_cfg["projector"]["in_dim"] = int(dim)

                    if "kge" not in align_cfg or not isinstance(align_cfg.get("kge"), dict):
                        align_cfg["kge"] = {}
                    align_cfg["kge"]["model"] = model
                    align_cfg["kge"]["dim"] = int(dim)

                    force_safe_dataloader_cfg(align_cfg)

                    align_cfg_path = cfg_dir / "align.yaml"
                    dump_yaml(align_cfg, align_cfg_path)

                    align_out = Path(deep_get(align_cfg, ["run", "out_dir"]))
                    align_ckpt = expected_best_ckpt(align_out)

                    # -------------------
                    # 3) Eval config
                    # -------------------
                    eval_cfg = copy.deepcopy(base_eval)
                    deep_set(eval_cfg, ["seed"], seed)
                    deep_set(eval_cfg, ["device"], args.device)
                    deep_set(eval_cfg, ["dtype"], args.dtype)
                    deep_set(eval_cfg, ["data", "data_dir"], str(data_dir))
                    deep_set(eval_cfg, ["run", "out_dir"], str(run_dir / "eval"))
                    deep_set(eval_cfg, ["kge_ckpt"], str(kge_ckpt))
                    deep_set(eval_cfg, ["align_ckpt"], str(align_ckpt))

                    if deep_get(eval_cfg, ["eval", "topk"], None) is None:
                        deep_set(eval_cfg, ["eval", "topk"], [1, 3, 5, 10, 20, 50])
                    if deep_get(eval_cfg, ["eval", "faiss"], None) is None:
                        deep_set(eval_cfg, ["eval", "faiss"], True)

                    force_safe_dataloader_cfg(eval_cfg)

                    eval_cfg_path = cfg_dir / "eval.yaml"
                    dump_yaml(eval_cfg, eval_cfg_path)

                    eval_out = Path(deep_get(eval_cfg, ["run", "out_dir"]))
                    eval_metrics_path = eval_out / "metrics_retrieval.json"

                    print(f"\n====================\n[RUN] {tag}\n====================")

                    # ---- KGE ----
                    if args.skip_kge:
                        print("[SKIP] KGE step by flag (--skip_kge).")
                    else:
                        if (not args.force) and kge_ckpt.exists():
                            print(f"[RESUME] KGE checkpoint exists -> {kge_ckpt}")
                        else:
                            run_cmd([py, "scripts/train_kge.py", "--config", str(kge_cfg_path)], dry_run=args.dry_run)

                    if not args.dry_run and not kge_ckpt.exists():
                        raise FileNotFoundError(f"KGE best checkpoint not found: {kge_ckpt}")

                    # ---- Align ----
                    if args.skip_align:
                        print("[SKIP] Align step by flag (--skip_align).")
                    else:
                        if (not args.force) and align_ckpt.exists():
                            print(f"[RESUME] Align checkpoint exists -> {align_ckpt}")
                        else:
                            run_cmd([py, "scripts/train_align.py", "--config", str(align_cfg_path)], dry_run=args.dry_run)

                    if not args.dry_run and not align_ckpt.exists():
                        raise FileNotFoundError(f"Align best checkpoint not found: {align_ckpt}")

                    # ---- Eval ----
                    if args.skip_eval:
                        print("[SKIP] Eval step by flag (--skip_eval).")
                    else:
                        if (not args.force) and eval_metrics_path.exists():
                            print(f"[RESUME] Eval metrics exists -> {eval_metrics_path}")
                        else:
                            run_cmd([py, "scripts/eval_retrieval.py", "--config", str(eval_cfg_path)], dry_run=args.dry_run)

    print("\n[DONE] All benchmark runs finished.")


if __name__ == "__main__":
    main()
