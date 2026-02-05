import torch

def get_device(name: str) -> torch.device:
    if name == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)

def get_dtype(dtype: str):
    if dtype.lower() in ["float64", "double"]:
        return torch.float64
    return torch.float32
