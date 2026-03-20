import json
import random
from pathlib import Path

import numpy as np
import torch


def load_config(path: str = "configs/default.json") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def get_device(cfg: dict) -> str:
    want = cfg.get("device", "cuda")
    if want == "cuda" and torch.cuda.is_available():
        return "cuda"
    return "cpu"