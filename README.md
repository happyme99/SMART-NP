# SMART-NP

Current research pipeline, reduced to its working core:

1. retrieve low-confidence natural-product queries with backbone embeddings;
2. select per-class RDKit descriptors ranked by mutual information;
3. ask a local GGUF LLM through `llama.cpp`;
4. optionally mark high-MSP predictions with MSP local FDR.

## One-time setup

```bash
./scripts/build_env.sh    # uv environment + packages only
./scripts/build_model.sh
```

This installs `uv` if needed, then runs `uv sync` to create `SMART-NP/` and install project packages, downloads default `qwen3.5-9b` GGUF under `models/`, and builds CUDA `llama.cpp` under `.tools/llama.cpp`.
Later `scripts/run.sh` calls detect both files and print `ready; build and download skipped`.

Use another registry model with `./scripts/build_model.sh --model NAME`. Add it to `configs/models.json`; or supply an ad-hoc model:

```bash
./scripts/build_model.sh --model my-model --model-repo org/repo --model-file model.gguf
naturalrag run --train-table train.tsv --train-embeddings train.pt --test-table test.tsv --test-embeddings test.pt --test-softmax test_softmax.pt --val-table val.tsv --val-softmax val_softmax.pt --model my-model
```

## Run GIN random data

```bash
./scripts/run.sh \
  --train-table data/dataset/random_train.tsv --train-embeddings data/dataset/gin_random_train_emb.pt \
  --test-table data/dataset/random_test.tsv --test-embeddings data/dataset/gin_random_test_emb.pt --test-softmax data/dataset/gin_random_test_softmax.pt \
  --calibration-table data/dataset/gin_random_calibration.tsv --calibration-softmax data/dataset/gin_random_calibration_softmax.pt
```

First run shows descriptor-rank tqdm and writes `data/descriptor-ranks/descriptor_selection_results.tsv`; later runs report the cache and reuse it. A 3-stage LFDR tqdm normalizes MSP with val, estimates test local FDR using `statsmodels`, and saves test rows with `lfdr > 0.05` to a softmax-specific file under `data/uncertain/`.
