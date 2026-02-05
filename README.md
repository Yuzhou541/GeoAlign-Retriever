# XGeo-KGRAG: Cross-Geometry Alignment for KG Retrieval (PyTorch)


> Align **non-Euclidean KG structural embeddings** (e.g., hyperbolic/Poincaré) into the **Euclidean LLM semantic space** for **ANN-friendly retrieval**, while measuring the **structure–efficiency trade-off**.

It includes:
- **KGE training**: Euclidean (TransE) + Hyperbolic (Poincaré ball via `geoopt`)
- **Alignment training**: learn a **projector** from manifold/tangent space → Euclidean (LLM/text) embedding space
- **Retrieval eval**: brute-force (GPU) + FAISS (CPU) + latency/memory metrics
- **Distortion metrics**: neighborhood consistency + graph-distance correlation after projection
- A **toy KG dataset** to verify end-to-end.

---

## 0) Prereqs (Windows + RTX4090 + CUDA 12.8)

- Windows 10/11
- Latest NVIDIA driver (supports CUDA 12.8 runtime)
- Conda (Miniconda/Anaconda)
- VSCode (recommended: Python extension)

PyTorch supports CUDA 12.8 wheels via the `cu128` index.

---

## 1) Create conda env (Python 3.10)

Open **VSCode → Terminal** (PowerShell recommended), then:

```powershell
conda create -n xgeo_kgrag python=3.10 -y
conda activate xgeo_kgrag

python -m pip install --upgrade pip
```

### 1.1 Install PyTorch (CUDA 12.8)

```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

Verify:

```powershell
python -c "import torch; print(torch.__version__); print('cuda?', torch.cuda.is_available()); print('cuda_ver', torch.version.cuda); print('gpu', torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"
```

### 1.2 Install other deps

```powershell
pip install -r requirements.txt
```

Then install this repo as an editable package (so `scripts/` can import `xgeo_kgrag`):

```powershell
pip install -e .
```


> Note: `geoopt` can be installed from PyPI (`pip install geoopt`) or from GitHub for the newest features.
> We default to PyPI in `requirements.txt`.

---

## 2) Quickstart (toy data)

### 2.1 Create toy KG (already included, but you can regenerate)

```powershell
python scripts/make_toy_kg.py --out_dir data/toykg
```

### 2.2 Train KGE (Hyperbolic Poincaré)

```powershell
python scripts/train_kge.py --config configs/kge_poincare.yaml
```

### 2.3 Train Align Projector (structure → text space)

```powershell
python scripts/train_align.py --config configs/align_toy.yaml
```

### 2.4 Evaluate retrieval

```powershell
python scripts/eval_retrieval.py --config configs/eval_toy.yaml
```

Outputs (checkpoints, logs, metrics) go to `runs/`.

---

## 3) Use your own KG

Put your triples into:

```
data/<your_kg>/
  train.txt
  valid.txt
  test.txt
```

Each line format:

```
head<TAB>relation<TAB>tail
```

Then point configs:
- `data_dir: data/<your_kg>`
- `kge.model: poincare` or `transe`

To train aligner on real text, provide:

```
data/<your_kg>/entity_texts.tsv
```

Format:

```
entity<TAB>text
```

---

## 4) Project structure

```
xgeo_kgrag/
  configs/
  data/toykg/
  scripts/
  src/xgeo_kgrag/
```

---

## License

MIT
