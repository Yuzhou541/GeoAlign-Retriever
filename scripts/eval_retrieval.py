import argparse
from pathlib import Path
import yaml

from xgeo_kgrag.utils.seed import set_seed
from xgeo_kgrag.utils.io import ensure_dir, save_json
from xgeo_kgrag.eval.retrieval import eval_retrieval


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    args = ap.parse_args()

    cfg_path = Path(args.config)
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

    set_seed(cfg.get("seed", 42))

    # output dir
    out_dir = Path(cfg.get("run", {}).get("out_dir", "runs/eval_tmp"))
    ensure_dir(out_dir)
    (out_dir / "configs").mkdir(exist_ok=True, parents=True)
    (out_dir / "configs" / cfg_path.name).write_text(cfg_path.read_text(encoding="utf-8"), encoding="utf-8")

    metrics = eval_retrieval(cfg)
    save_json(out_dir / "metrics_retrieval.json", metrics)
    print(f"[DONE] Retrieval eval complete. Saved -> {out_dir / 'metrics_retrieval.json'}")


if __name__ == "__main__":
    main()
