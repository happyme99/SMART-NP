"""Embedding retrieval shared by every backbone."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class RetrievedExample:
    rank: int
    train_idx: int
    class_name: str
    smiles: str
    similarity: float


class TrainIndex:
    def __init__(self, frame: pd.DataFrame, vectors: torch.Tensor):
        if len(frame) != len(vectors):
            raise ValueError(f"Train rows ({len(frame)}) != embeddings ({len(vectors)})")
        self.frame, self.vectors = frame.reset_index(drop=True), vectors.float().cpu()

    def _score(self, query: torch.Tensor, vectors: torch.Tensor | None = None) -> torch.Tensor:
        library = self.vectors if vectors is None else vectors
        query = query.float().view(1, -1)
        return F.cosine_similarity(query, library, dim=1)

    def topk(self, query: torch.Tensor, k: int) -> list[RetrievedExample]:
        scores, indices = torch.topk(self._score(query), k=min(k, len(self.frame)), largest=True, sorted=True)
        return [self._row(rank, int(index), float(score)) for rank, (score, index) in enumerate(zip(scores.tolist(), indices.tolist()), 1)]

    def top1_for_class(self, query: torch.Tensor, class_name: str) -> RetrievedExample | None:
        mask = self.frame["true_class"].astype(str).str.strip().eq(str(class_name).strip()).to_numpy()
        if not mask.any():
            return None
        indices = torch.as_tensor(mask.copy()).nonzero().flatten()
        scores = self._score(query, self.vectors[indices])
        position = int(scores.argmax())
        index = int(indices[position])
        return self._row(0, index, float(scores[position]))

    def _row(self, rank: int, index: int, similarity: float) -> RetrievedExample:
        row = self.frame.iloc[index]
        return RetrievedExample(rank=rank, train_idx=index, class_name=str(row["true_class"]).strip(), smiles=str(row["SMILES"]).strip(), similarity=similarity)


def prompt_examples(retrieved: list[RetrievedExample], baseline: RetrievedExample | None, baseline_label: str, topk: int) -> list[RetrievedExample]:
    primary = retrieved[:topk]
    covered = next((item for item in primary if item.class_name == baseline_label), None)
    if covered is not None:
        examples = primary
    else:
        examples = primary + ([baseline] if baseline is not None else [])
    return [RetrievedExample(rank=index, train_idx=item.train_idx, class_name=item.class_name, smiles=item.smiles, similarity=item.similarity) for index, item in enumerate(examples, 1)]
