"""Command line entrypoint."""

from __future__ import annotations

import argparse
import subprocess
from datetime import datetime
from pathlib import Path

from model_setup import resolve_model, setup_model
from runner import run


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="naturalrag")
    commands = root.add_subparsers(dest="command", required=True)

    setup = commands.add_parser("build-model", help="download GGUF and build llama.cpp")
    setup.add_argument("--model", default="qwen3.5-9b")
    setup.add_argument("--model-repo")
    setup.add_argument("--model-file")
    setup.add_argument("--model-revision", default="main")
    setup.add_argument("--llama-ref", default="master")
    setup.add_argument("--cpu", action="store_true", help="build CPU llama.cpp instead of CUDA")

    inference = commands.add_parser("run", help="run descriptor-selected RAG from explicit files")
    inference.add_argument("--train-table", required=True, type=Path)
    inference.add_argument("--train-embeddings", required=True, type=Path)
    inference.add_argument("--test-table", required=True, type=Path)
    inference.add_argument("--test-embeddings", required=True, type=Path)
    inference.add_argument("--test-softmax", required=True, type=Path)
    inference.add_argument("--calibration-table", "--val-table", dest="calibration_table", required=True, type=Path)
    inference.add_argument("--calibration-softmax", "--val-softmax", dest="calibration_softmax", required=True, type=Path)
    inference.add_argument("--model", default="qwen3.5-9b")
    inference.add_argument("--model-path", type=Path)
    inference.add_argument("--output", type=Path)
    inference.add_argument("--uncertain-output", type=Path, help="path for the MSP--LFDR uncertain-sample cache")
    inference.add_argument("--topk", type=int, default=4)
    inference.add_argument("--descriptor-topk", type=int, default=4)
    inference.add_argument("--max-queries", type=int)
    inference.add_argument("--start-index", type=int, default=0)
    inference.add_argument("--port", type=int, default=8091)
    inference.add_argument("--cuda-device", type=int)
    inference.add_argument("--threads", type=int, default=32)
    inference.add_argument("--ctx", type=int, default=8191)
    inference.add_argument("--resume", action="store_true")

    return root


def main(argv: list[str] | None = None) -> None:
    args = parser().parse_args(argv)
    if args.command == "build-model":
        setup_model(resolve_model(args.model, args.model_repo, args.model_file, args.model_revision), args.llama_ref, cuda=not args.cpu)
    elif args.command == "run":
        uv_version = subprocess.run(["uv", "--version"], capture_output=True, check=True, text=True).stdout.strip()
        print(f"{'[uv environment exists]':<41}  {'':<12}  {uv_version}", flush=True)
        if args.model_path is None:
            setup_model(resolve_model(args.model))
        created_at = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        output = args.output or Path("results") / f"{args.model}_{created_at}_results.tsv"
        run(train_table=args.train_table, train_embeddings=args.train_embeddings, test_table=args.test_table, test_embeddings=args.test_embeddings, test_softmax=args.test_softmax, calibration_table=args.calibration_table, calibration_softmax=args.calibration_softmax, model_name=args.model, model_file=args.model_path, output=output, uncertain_output=args.uncertain_output, topk=args.topk, descriptor_topk=args.descriptor_topk, max_queries=args.max_queries, start_index=args.start_index, port=args.port, cuda_device=args.cuda_device, threads=args.threads, ctx=args.ctx, resume=args.resume)
