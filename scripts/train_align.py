import argparse
from pathlib import Path
import yaml

from xgeo_kgrag.utils.seed import set_seed
from xgeo_kgrag.utils.io import ensure_dir, save_json
from xgeo_kgrag.train.align_trainer import train_align

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    set_seed(cfg.get("seed", 42))

    out_dir = Path(cfg["run"]["out_dir"])
    ensure_dir(out_dir)
    (out_dir / "configs").mkdir(exist_ok=True, parents=True)
    (out_dir / "configs" / Path(args.config).name).write_text(Path(args.config).read_text(encoding="utf-8"), encoding="utf-8")

    metrics = train_align(cfg)
    save_json(out_dir / "metrics_final.json", metrics)
    print("[DONE] Alignment training complete.")

if __name__ == "__main__":
    main()
