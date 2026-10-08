# SMART-NP

SMART-NP refines natural-product class predictions for test samples whose classifier confidence is uncertain. It uses MSP local FDR (LFDR) to route samples, retrieves training examples by backbone embedding, selects class-specific RDKit descriptors by mutual information, and asks a local GGUF LLM for a class label.

![SMART-NP full pipeline](full_pipeline.png)

The diagram shows the full routing design. This repository's `run` command writes results **only for samples routed to the LLM** (`LFDR > 0.05`). It does not merge classifier predictions for accepted samples (`LFDR <= 0.05`) into the result file.

## Repository layout

| Path | Contents |
| --- | --- |
| `src/` | CLI, LFDR routing, embedding retrieval, descriptor ranking, prompt construction, and LLM inference |
| `scripts/` | Environment, model, and run commands |
| `configs/models.json` | GGUF model registry; default: `qwen3.5-9b` |
| `data/README.md` | Input layout and example paths |
| `SMART-NP/`, `.tools/`, `models/`, `data/dataset/`, `data/descriptor-ranks/`, `data/uncertain/`, `results/` | Local environment, downloads, inputs, caches, and outputs (ignored by Git) |

## Setup

Run from the repository root. Setup needs `git`, `cmake`, `uv` (installed by `build_env.sh` if missing), and CUDA with `nvcc` for the `llama.cpp` build.

```bash
./scripts/build_env.sh
./scripts/build_model.sh
```

`build_env.sh` creates `SMART-NP/` and installs packages from `uv.lock`. `build_model.sh` downloads the default GGUF into `models/qwen3.5-9b/` and builds `llama-server` in `.tools/llama.cpp/`. Existing model and server files are reused.

For another model, add it to `configs/models.json` and pass `--model NAME` to `build_model.sh` and `run.sh`. To download an unregistered model, supply its local path when running:

```bash
./scripts/build_model.sh --model my-model --model-repo org/repo --model-file model.gguf
# Add --model my-model --model-path models/my-model/model.gguf to the run command below.
```

## Run with the local GIN example

Input tables need `SMILES` and `true_class` columns. Embedding `.pt` files need a 2D tensor in table row order; softmax `.pt` files need `probabilities` and `class_to_idx`. Train, test, and calibration files must use matching rows within each split. Large inputs under `data/dataset/` are local files, not included in Git.

```bash
./scripts/run.sh \
  --train-table data/dataset/random_train.tsv \
  --train-embeddings data/dataset/gin_random_train_emb.pt \
  --test-table data/dataset/random_test.tsv \
  --test-embeddings data/dataset/gin_random_test_emb.pt \
  --test-softmax data/dataset/gin_random_test_softmax.pt \
  --calibration-table data/dataset/gin_random_calibration.tsv \
  --calibration-softmax data/dataset/gin_random_calibration_softmax.pt
```

The first run ranks descriptors into `data/descriptor-ranks/descriptor_selection_results.tsv` and saves routed test rows in `data/uncertain/gin_random_test_softmax_lfdr_uncertain_samples.tsv`. Later runs reuse these files. LLM predictions go to a timestamped TSV under `results/` with `SMILES`, `true`, `backbone_pred`, and `llm_pred`; printed accuracy covers those routed rows only. Use `--output PATH` for a fixed result path and `--resume` to continue that file.
