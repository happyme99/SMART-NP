"""Feature-selected RAG inference from explicit input files."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

from model_setup import ROOT, installed_model_path
from confidence import annotate_msp_lfdr
from data import embeddings, softmax_results, train_frame
from descriptor import DescriptorStore, rank_descriptors
from llm import LlamaServer, extract_label
from prompt import build_prompt
from retrieval import TrainIndex, prompt_examples

RESULT_COLUMNS = ["SMILES", "true", "backbone_pred", "llm_pred"]


def run(
    *,
    train_table: Path,
    train_embeddings: Path,
    test_table: Path,
    test_embeddings: Path,
    test_softmax: Path,
    calibration_table: Path,
    calibration_softmax: Path,
    model_name: str,
    model_file: Path | None,
    output: Path,
    uncertain_output: Path | None = None,
    topk: int = 4,
    descriptor_topk: int = 4,
    max_queries: int | None = None,
    start_index: int = 0,
    port: int = 8091,
    cuda_device: int | None = None,
    threads: int = 32,
    ctx: int = 8191,
    resume: bool = False,
) -> dict[str, object]:
    train, train_vectors = train_frame(train_table), embeddings(train_embeddings)
    uncertain_path = uncertain_output or ROOT / "data/uncertain" / f"{test_softmax.stem}_lfdr_uncertain_samples.tsv"
    
    if uncertain_path.is_file():
        query, query_vectors = train_frame(uncertain_path), embeddings(test_embeddings)
        if "source_row_idx" in query:
            source_indices = pd.to_numeric(query["source_row_idx"], errors="raise").to_numpy(dtype="int64", copy=True)
        else:
            source = train_frame(test_table)
            if source["SMILES"].duplicated().any():
                raise ValueError("LFDR cache without source_row_idx cannot use duplicate test SMILES")
            source_indices = query["SMILES"].map(pd.Series(source.index, index=source["SMILES"])).to_numpy(dtype="int64", copy=True)
        if len(query_vectors) != len(train_frame(test_table)) or (source_indices < 0).any() or (source_indices >= len(query_vectors)).any():
            raise ValueError(f"Invalid LFDR cache: {uncertain_path}")
        query_vectors = query_vectors[torch.tensor(source_indices, dtype=torch.long)]
        print(f"{'[Uncertain sample selection already done]':<41}  {f'samples={len(query)}':<12}  path={uncertain_path}", flush=True)
    else:
        with tqdm(total=3, desc="LFDR uncertainty", unit="stage") as progress:
            calibration = _classification(calibration_table, calibration_softmax)
            progress.update()
            query, query_vectors = _query(test_table, test_embeddings, test_softmax)
            progress.update()
            annotated = annotate_msp_lfdr(calibration, query, q=0.05)
            progress.update()
        uncertain = annotated.loc[~annotated["accept"]].copy()
        uncertain.insert(0, "source_row_idx", uncertain.index)
        uncertain_path.parent.mkdir(parents=True, exist_ok=True)
        uncertain.to_csv(uncertain_path, sep="\t", index=False)
        query_vectors = query_vectors[torch.tensor(uncertain["source_row_idx"].to_numpy(), dtype=torch.long)]
        query = uncertain.reset_index(drop=True)
        print(f"LFDR uncertain\tsamples={len(query)}/{len(annotated)}\tlfdr_limit=0.05\tsaved={uncertain_path}", flush=True)
    split = next((part for part in train_table.stem.split("_") if part in {"random", "scaffold", "logp"}), train_table.stem)
    descriptor_table = ROOT / "data/descriptor-ranks" / f"{split}_descriptor_selection_results.tsv"
    
    if not descriptor_table.is_file():
        rank_descriptors(train, descriptor_table, topk=max(6, descriptor_topk))
    else:
        print(f"{'[Descriptor ranking already done]':<41}  {'':<12}  path={descriptor_table}", flush=True)
    index, descriptor_store = TrainIndex(train, train_vectors), DescriptorStore(descriptor_table, topk=descriptor_topk)
    labels = sorted(train["true_class"].astype(str).str.strip().unique(), key=len, reverse=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not len(query):
        summary = {"output": str(output), "rows": 0, "accuracy": None}
        print(json.dumps(summary, indent=2), flush=True)
        return summary
    if resume and start_index:
        raise ValueError("--resume requires --start-index 0; result files do not store row indices")
    rows = pd.read_csv(output, sep="\t", low_memory=False).fillna("").to_dict("records") if resume and output.is_file() else []
    if rows and set(rows[0]) != set(RESULT_COLUMNS):
        raise ValueError(f"Cannot resume incompatible result file: {output}")
    completed = set(range(len(rows)))
    indices = range(max(0, start_index), len(query) if max_queries is None else min(len(query), start_index + max_queries))
    pending_indices = [row_idx for row_idx in indices if row_idx not in completed]
    position_width = max(3, len(str(len(pending_indices))))
    model = Path(model_file) if model_file else installed_model_path(model_name)
    binary = ROOT / ".tools/llama.cpp/build/bin/llama-server"
    with LlamaServer(binary, model, port, ctx=ctx, threads=threads, cuda_device=cuda_device) as client:
        for position, row_idx in enumerate(pending_indices, 1):
            item = query.iloc[row_idx]
            retrieved = index.topk(query_vectors[row_idx], topk)
            baseline_label = str(item["pred_class"]).strip()
            baseline = index.top1_for_class(query_vectors[row_idx], baseline_label)
            examples = prompt_examples(retrieved, baseline, baseline_label, topk)
            selected_descriptors = descriptor_store.select([baseline_label, *(example.class_name for example in examples)])
            system, user = build_prompt(str(item["SMILES"]), examples, selected_descriptors)
            raw, finish_reason = client.chat(system, user)
            prediction = extract_label(raw, labels)
            if finish_reason == "length" or prediction == "INVALID":
                raw, _ = client.chat(system, user, thinking=False)
                prediction = extract_label(raw, labels)
            record = {
                "SMILES": item["SMILES"],
                "true": item["true_class"],
                "backbone_pred": baseline_label,
                "llm_pred": prediction,
            }
            rows.append(record)
            pd.DataFrame(rows, columns=RESULT_COLUMNS).to_csv(output, sep="\t", index=False)
            tqdm.write(f"[{position:0{position_width}d}/{len(pending_indices):0{position_width}d}]\tSMILES={record['SMILES']}\tTrue_class={record['true']}\tLLM_pred={prediction}")
    frame = pd.DataFrame(rows)
    summary = {"output": str(output), "rows": len(frame), "accuracy": float(frame["llm_pred"].eq(frame["true"]).mean()) if len(frame) else None}
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def _query(table: Path, embedding: Path, softmax: Path) -> tuple[pd.DataFrame, torch.Tensor]:
    frame, vectors = _classification(table, softmax), embeddings(embedding)
    if len(frame) != len(vectors):
        raise ValueError(f"Query sizes differ: table={len(frame)}, embeddings={len(vectors)}")
    return frame, vectors


def _classification(table: Path, softmax: Path) -> pd.DataFrame:
    frame, results = train_frame(table), softmax_results(softmax)
    if len(frame) != len(results):
        raise ValueError(f"Classification sizes differ: table={len(frame)}, softmax={len(results)}")
    frame[["pred_class", "msp"]] = results
    frame["correct"] = frame["pred_class"].eq(frame["true_class"])
    return frame
