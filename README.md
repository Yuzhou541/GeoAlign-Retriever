# Geo-Scale Sweep Final Artifacts (A800 80GB)

### Project overview (GeoScale Sweep)

This project studies **how strongly “geometry” should influence entity ranking in KG benchmarks**. In many KG pipelines (including KG-augmented retrieval and embedding-based ranking), geometric structure is helpful—but only if it’s **calibrated**. Too little geometry reduces the model to a near-standard embedding scorer; too much geometry can distort neighborhoods and hurt ranking stability.

To make this calibration explicit, we introduce and benchmark a single, interpretable knob:

**`geo_scale`** — a scalar that controls the **strength of the geometry-driven signal** in scoring (e.g., the influence of distance/curvature-aware structure relative to the base embedding score).

---

### Core idea

Instead of claiming one geometry setting works universally, we treat geometry as a **tunable inductive bias** and do a controlled sweep:

* **Benchmarks:** FB15k-237, WN18RR, YAGO3-10
* **Backbones:** Poincaré (hyperbolic) and TransE (translation-based)
* **Dims:** 32 / 64
* **Metrics (mean±std over runs):** MRR, Hits@K, plus geometry diagnostics like **distance–rank correlation** and **neighborhood consistency**

We then close the experimental loop by:

1. **Selecting top-1 `geo_scale` per (kg, model, dim)** based on MRR (with deterministic tie-breaking).
2. Producing a **final summary table + main figure** using only these top-1 settings, plus a **delta-vs-baseline (`geo_scale=0`)** report for clean paper-ready comparisons.

This run directory contains the **geo_scale sweep** results and the **fully closed-loop** post-processing pipeline:
1) aggregate mean/std over repeated runs,
2) select **top-1 geo_scale per (kg, model, dim)** by **MRR mean**,
3) generate **final tables + paper-ready plots**.

> **Hardware used:** NVIDIA **A800 80GB** (80G HBM).  
> Plotting/aggregation is CPU-light; GPU is mainly for upstream training/eval.

---

## Directory layout (relevant)

- `runs/kdd_bench/q2e_geoscale_sweep.csv`  
  Raw sweep results (may include an accidental duplicated header row inside the file).
- `runs/kdd_bench/q2e_geoscale_sweep.clean.csv`  
  Cleaned version (header-row artifacts removed).
- `runs/kdd_bench/q2e_geoscale_sweep_mean_std.recalc.csv`  
  Aggregated mean/std over runs, grouped by `(kg, model, dim, geo_scale)`.
- `runs/kdd_bench/q2e_geoscale_best_by_mrr.csv`  
  **Top-1** row per `(kg, model, dim)` selected by `mrr_mean`.
- `runs/kdd_bench/figs/make_geoscale_final.py`  
  Generates **final summary tables** + **paper plots** from sweep + best.

Outputs written to `runs/kdd_bench/`:
- `q2e_geoscale_best_pivot_fmt.csv` (paper-friendly formatted)
- `q2e_geoscale_best_pivot_num.csv` (pure numeric)
- `q2e_geoscale_delta_vs_gs0.csv` (best − geo_scale=0 baseline deltas)
- `geoscale_sweep_mrr_all.pdf` (sweep curves)
- `best_mrr_bar.pdf` (best MRR bar chart)

---

## Dependencies

### 1) Miller (`mlr`)
Used for CSV cleaning, aggregation, and top-1 selection.

```bash
mlr --version
```

### 2) Python plotting env (IMPORTANT: avoid NumPy 2.x issues)
Some base images ship with **NumPy 2.x**, which can break Matplotlib/tight_layout in certain environments.
Pin a clean env for figures:

```bash
conda create -y -n geofigs python=3.10 numpy=1.26.4 pandas matplotlib=3.8.4
conda run -n geofigs python -c "import numpy; import numpy.core.umath as u; print('OK', numpy.__version__, hasattr(u,'ERR_IGNORE'))"
```

---

## Reproduce the full “experiment closure” (copy-paste)

From `runs/kdd_bench/figs/`:

```bash
cd /root/xgeo_kgrag/runs/kdd_bench/figs
```

### Step 0 — Clean raw CSV (remove accidental header rows)

```bash
mlr --csv filter '$kg != "kg"' ../q2e_geoscale_sweep.csv > ../q2e_geoscale_sweep.clean.csv
```

### Step 1 — Aggregate mean/std per (kg, model, dim, geo_scale)

```bash
mlr --csv stats1 -a mean,stddev \
  -f mrr,"hits@1","hits@10","hits@100","hits@1000",dist_rank_corr,neigh_consistency \
  -g kg,model,dim,geo_scale \
  ../q2e_geoscale_sweep.clean.csv \
  > ../q2e_geoscale_sweep_mean_std.recalc.csv
```

### Step 2 — Select **top-1 geo_scale** per (kg, model, dim) by MRR mean

We sort within each `(kg,model,dim)` by descending `mrr_mean`, then keep the first occurrence.

```bash
mlr --csv \
  sort -f kg,model -n dim -nr mrr_mean then \
  put 'k=$kg."|".$model."|".$dim; $keep = haskey(@seen,k) ? 0 : 1; @seen[k]=1' then \
  filter '$keep==1' then \
  cut -x -f keep \
  ../q2e_geoscale_sweep_mean_std.recalc.csv \
> ../q2e_geoscale_best_by_mrr.csv
```

### Step 3 — Sanity check (must be exactly 12 groups)

Expected groups: `3 KG × 2 model × 2 dim = 12`.

```bash
mlr --csv count-distinct -f kg,model,dim ../q2e_geoscale_best_by_mrr.csv
# Expected: 12 lines, each count=1
```

Optional: inspect one group and confirm top-1 is consistent with sweep:

```bash
mlr --csv filter '$kg=="fb15k237" && $model=="poincare" && $dim==64' ../q2e_geoscale_sweep_mean_std.recalc.csv \
| mlr --csv sort -nr mrr_mean \
| mlr --csv head -n 3

mlr --csv filter '$kg=="fb15k237" && $model=="poincare" && $dim==64' ../q2e_geoscale_best_by_mrr.csv
```

### Step 4 — Generate final paper artifacts (tables + plots)

Run in the pinned plotting env:

```bash
conda run -n geofigs python3 make_geoscale_final.py \
  --sweep_csv ../q2e_geoscale_sweep_mean_std.recalc.csv \
  --best_csv  ../q2e_geoscale_best_by_mrr.csv \
  --outdir .. \
  --metric mrr_mean
```

You should see:
- `q2e_geoscale_best_pivot_fmt.csv`
- `q2e_geoscale_best_pivot_num.csv`
- `q2e_geoscale_delta_vs_gs0.csv`
- `geoscale_sweep_mrr_all.pdf`
- `best_mrr_bar.pdf`

---

## What to cite/use in the paper

**Main table:**
- `q2e_geoscale_best_pivot_fmt.csv` (copy into LaTeX/Markdown table)

**Ablation / relative gain vs baseline geo_scale=0:**
- `q2e_geoscale_delta_vs_gs0.csv`

**Main plots:**
- `geoscale_sweep_mrr_all.pdf` (MRR vs geo_scale curves, all groups)
- `best_mrr_bar.pdf` (best-by-group comparison)

Notes:
- `*_mean` and `*_stddev` are computed from repeated runs (seeds / repeats).

---

## Export artifacts from server to local

### Option A: tarball (recommended)

On server:

```bash
cd /root/xgeo_kgrag/runs/kdd_bench
tar -czf geoscale_final_artifacts.tgz \
  q2e_geoscale_best_by_mrr.csv \
  q2e_geoscale_best_pivot_fmt.csv \
  q2e_geoscale_best_pivot_num.csv \
  q2e_geoscale_delta_vs_gs0.csv \
  geoscale_sweep_mrr_all.pdf \
  best_mrr_bar.pdf \
  q2e_geoscale_sweep_mean_std.recalc.csv
```

On local:

```bash
scp USER@HOST:/root/xgeo_kgrag/runs/kdd_bench/geoscale_final_artifacts.tgz .
tar -xzf geoscale_final_artifacts.tgz
```

### Option B: scp individual files

```bash
scp USER@HOST:/root/xgeo_kgrag/runs/kdd_bench/q2e_geoscale_best_by_mrr.csv .
scp USER@HOST:/root/xgeo_kgrag/runs/kdd_bench/q2e_geoscale_best_pivot_fmt.csv .
scp USER@HOST:/root/xgeo_kgrag/runs/kdd_bench/q2e_geoscale_delta_vs_gs0.csv .
scp USER@HOST:/root/xgeo_kgrag/runs/kdd_bench/geoscale_sweep_mrr_all.pdf .
scp USER@HOST:/root/xgeo_kgrag/runs/kdd_bench/best_mrr_bar.pdf .
```

---

## Troubleshooting

### “object __array__ method not producing an array” / NumPy import errors

Cause: incompatible NumPy/Matplotlib stack (often NumPy 2.x in some containers).

Fix: use the pinned env:

```bash
conda run -n geofigs python3 make_geoscale_final.py ...
```

### `mlr group-by` usage confusion

`mlr group-by` syntax is:

```bash
mlr group-by a,b,c input.csv
```

In this pipeline we avoid `group-by` and use explicit “seen/dedup” logic for top-1 selection.

---

## Done criteria (“experiment closed loop”)

- [x] `q2e_geoscale_sweep_mean_std.recalc.csv` generated
- [x] `q2e_geoscale_best_by_mrr.csv` has **exactly one row** per `(kg,model,dim)` (12 groups)
- [x] final tables + plots generated (`*_pivot_*`, `delta_vs_gs0`, PDFs)
