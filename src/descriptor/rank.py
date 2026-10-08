"""Class-specific RDKit descriptor ranks and prompt-time union selection."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mutual_info_score
from tqdm import tqdm

from .rdkit import DESCRIPTOR_KEYS, DESCRIPTOR_LABELS, compute_descriptors


class DescriptorStore:
    def __init__(self, path: Path, topk: int = 4):
        if not path.is_file():
            raise FileNotFoundError(f"Descriptor rank table missing: {path}")
        frame = pd.read_csv(path, sep="\t", low_memory=False)
        required = {"class_name", "rank", "feature"}
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{path}: descriptor rank table missing {sorted(missing)}")
        frame["rank"] = pd.to_numeric(frame["rank"], errors="raise")
        self.rows = frame.loc[frame["rank"] <= int(topk)].sort_values(["class_name", "rank", "feature"], kind="stable")

    def select(self, classes: list[str]) -> list[str]:
        ordered = list(dict.fromkeys(str(name).strip() for name in classes if str(name).strip()))
        selected, seen = [], set()
        for class_name in ordered:
            for descriptor in self.rows.loc[self.rows["class_name"].eq(class_name), "feature"]:
                descriptor = str(descriptor).strip()
                if descriptor and descriptor not in seen:
                    seen.add(descriptor)
                    selected.append(descriptor)
        return selected


def rank_descriptors(train: pd.DataFrame, output: Path, topk: int = 6) -> Path:
    required = {"SMILES", "true_class"}
    missing = required.difference(train.columns)
    if missing:
        raise ValueError(f"Training table missing {sorted(missing)}")
    rows = [compute_descriptors(smiles) for smiles in tqdm(train["SMILES"], desc="RDKit descriptors", unit="molecule")]
    matrix = np.asarray([[float(value[key]) if not isinstance(value[key], list) else float(len(value[key])) for key in DESCRIPTOR_KEYS] for value in rows])
    labels = train["true_class"].astype(str).str.strip().to_numpy()
    class_names, class_codes = np.unique(labels, return_inverse=True)
    score = _mi_scores(matrix, class_codes, len(class_names))
    result = []
    for index, class_name in enumerate(class_names):
        target = class_codes == index
        if target.sum() < 2 or (~target).sum() < 2:
            continue
        weighted_score = score[index] * target.mean()
        for rank, descriptor_index in enumerate(np.argsort(-weighted_score, kind="stable")[:topk], start=1):
            result.append({"class_name": class_name, "class_count": int(target.sum()), "class_ratio": float(target.mean()), "rank": rank, "feature": DESCRIPTOR_KEYS[descriptor_index], "feature_label": DESCRIPTOR_LABELS[DESCRIPTOR_KEYS[descriptor_index]], "weighted_mi_score": float(weighted_score[descriptor_index])})
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(result).to_csv(output, sep="\t", index=False)
    return output


def _mi_scores(matrix: np.ndarray, class_codes: np.ndarray, n_classes: int) -> np.ndarray:
    scores = np.empty((n_classes, matrix.shape[1]))
    for descriptor_index in tqdm(range(matrix.shape[1]), desc="Descriptor ranking", unit="descriptor"):
        _, descriptor_codes = np.unique(matrix[:, descriptor_index], return_inverse=True)
        n_values = int(descriptor_codes.max()) + 1
        counts = np.bincount(class_codes * n_values + descriptor_codes, minlength=n_classes * n_values).reshape(n_classes, n_values)
        totals = counts.sum(axis=0)
        for class_index in range(n_classes):
            scores[class_index, descriptor_index] = mutual_info_score(None, None, contingency=np.stack((counts[class_index], totals - counts[class_index])))
    return scores
