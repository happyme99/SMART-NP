"""Strict input loading."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch

def train_frame(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Missing data file: {path}")
    frame = pd.read_csv(path, sep="," if path.suffix == ".csv" else "\t", low_memory=False).fillna("")
    required = {"SMILES", "true_class"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path}: train table missing {sorted(missing)}")
    for column in required:
        frame[column] = frame[column].astype(str).str.strip()
    if frame["SMILES"].eq("").any():
        raise ValueError("SMILES contains an empty value")
    return frame


def embeddings(path: Path) -> torch.Tensor:
    if not path.is_file():
        raise FileNotFoundError(f"Missing embedding file: {path}")
    value = torch.load(path, map_location="cpu")
    if isinstance(value, dict):
        value = value.get("embeddings", value)
    if not isinstance(value, torch.Tensor) or value.ndim != 2:
        raise TypeError(f"{path}: expected a rank-2 torch.Tensor")
    return value.float().cpu()


def softmax_results(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Missing softmax file: {path}")
    value = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(value, dict):
        raise TypeError(f"{path}: expected a dict with probabilities and class_to_idx")
    probabilities, class_to_idx = value.get("probabilities"), value.get("class_to_idx")
    if not isinstance(probabilities, torch.Tensor) or probabilities.ndim != 2 or not isinstance(class_to_idx, dict):
        raise TypeError(f"{path}: expected rank-2 probabilities and class_to_idx")
    probabilities = probabilities.float()
    if not torch.isfinite(probabilities).all() or torch.any(probabilities.sum(dim=1) <= 0):
        raise ValueError(f"{path}: expected finite probability rows with positive sums")
    probabilities = probabilities / probabilities.sum(dim=1, keepdim=True)
    index_to_class = {int(index): str(class_name) for class_name, index in class_to_idx.items()}
    if set(index_to_class) != set(range(probabilities.shape[1])):
        raise ValueError(f"{path}: class_to_idx must cover every softmax column")
    return pd.DataFrame(
        {
            "pred_class": [index_to_class[index] for index in probabilities.argmax(dim=1).tolist()],
            "msp": probabilities.max(dim=1).values.numpy(),
        }
    )
