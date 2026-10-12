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
| `pyproject.toml`, `uv.lock` | Python dependencies, CLI entry point, and locked versions (tracked by Git) |
| `data/README.md` | Input layout and example paths |
| `SMART-NP/`, `.tools/`, `models/`, `data/descriptor-ranks/`, `data/uncertain/`, `results/` | Local environment, downloads, caches, and outputs (ignored by Git) |
| `data/dataset/` | Input directory included in Git; downloaded files remain ignored |

## Clone and download data

Clone the private repository with your GitHub credentials:

```bash
git clone https://github.com/happyme99/SMART-NP.git
cd SMART-NP
```

All commands below assume `uv` is available. If it is missing, install it once with the [official installer](https://docs.astral.sh/uv/getting-started/installation/):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

If `gh` is missing, install it using [Webi](https://github.com/cli/cli/blob/trunk/docs/install_linux.md#webi):

```bash
curl -sS https://webi.sh/gh | sh
```

Release assets are separate from Git, so the directory starts empty.

The example input files are distributed separately through the [data-v1 release](https://github.com/happyme99/SMART-NP/releases/tag/data-v1). Download and extract them into `data/dataset/` with:

```bash
uv run scripts/download_data.sh
```

The release contains 46 TSV/PT files: 11 each for MolFormer, GIN, NaFM, and NPClassifier, plus the shared `random_train.tsv` and `random_test.tsv`. MolFormer uses those shared tables for its train and test embeddings because it has no model-specific train/test tables in the archive. The GIN example below uses its own matching files.

## Setup

Choose the command for the step you want. Run it from the repository root:

| Goal | Command |
| --- | --- |
| Create the Python environment only | `./scripts/build_env.sh` |
| Create the environment, build `llama.cpp`, and download Qwen | `./scripts/build_model.sh` |
| Do the setup and run inference | `./scripts/run.sh` with the input files shown below |

`build_env.sh` uses `pyproject.toml` and `uv.lock` to create `SMART-NP/`. Keep both files in Git. `build_model.sh` also runs that step, then builds `.tools/llama.cpp/build/bin/llama-server` and downloads the default `Qwen3.5-9B-Q4_K_M.gguf` to `models/qwen3.5-9b/`. `run.sh` sets up the environment and, unless `--model-path` is supplied, also sets up the model before inference. Existing files are reused.

Requirements: Python 3.10+, `git`, `cmake`, and CUDA with `nvcc` for the default model build. For a CPU build, use `./scripts/build_model.sh --cpu` and pass `--model-path models/qwen3.5-9b/Qwen3.5-9B-Q4_K_M.gguf` to `run.sh`.

For another model, add it to `configs/models.json` and pass `--model NAME` to `build_model.sh` and `run.sh`. To download an unregistered model, supply its local path when running:

```bash
./scripts/build_model.sh --model my-model --model-repo org/repo --model-file model.gguf
# Add --model my-model --model-path models/my-model/model.gguf to the run command below.
```

## Run with the local GIN example

Input tables need `SMILES` and `true_class` columns. Embedding `.pt` files need a 2D tensor in table row order; softmax `.pt` files need `probabilities` and `class_to_idx`. Train, test, and calibration files must use matching rows within each split. The downloaded inputs under `data/dataset/` remain outside Git.

```bash
./scripts/run.sh \
  --train-table data/dataset/gin_random_train.tsv \
  --train-embeddings data/dataset/gin_random_train_emb.pt \
  --test-table data/dataset/gin_random_test.tsv \
  --test-embeddings data/dataset/gin_random_test_emb.pt \
  --test-softmax data/dataset/gin_random_test_softmax.pt \
  --calibration-table data/dataset/gin_random_calibration.tsv \
  --calibration-softmax data/dataset/gin_random_calibration_softmax.pt
```

The first run ranks descriptors into `data/descriptor-ranks/descriptor_selection_results.tsv` and saves routed test rows in `data/uncertain/gin_random_test_softmax_lfdr_uncertain_samples.tsv`. Later runs reuse these files. LLM predictions go to a timestamped TSV under `results/` with `SMILES`, `true`, `backbone_pred`, and `llm_pred`; printed accuracy covers those routed rows only. Use `--output PATH` for a fixed result path and `--resume` to continue that file.
