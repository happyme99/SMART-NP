# Data layout

Large inputs stay outside Git. Download them from the data-v1 release as shown in the [root README](../README.md), then pass each required file explicitly.

Pass explicit train/test/calibration files to `scripts/run.sh`:

```text
uv run scripts/run.sh \
  --train-table data/dataset/gin_random_train.tsv --train-embeddings data/dataset/gin_random_train_emb.pt \
  --test-table data/dataset/gin_random_test.tsv --test-embeddings data/dataset/gin_random_test_emb.pt --test-softmax data/dataset/gin_random_test_softmax.pt \
  --calibration-table data/dataset/gin_random_calibration.tsv --calibration-softmax data/dataset/gin_random_calibration_softmax.pt
```

First run automatically writes `data/descriptor-ranks/random_descriptor_selection_results.tsv`, then reuses it for the random split. A train table named `*_scaffold_train.tsv` uses `scaffold_descriptor_selection_results.tsv`; `*_logp_train.tsv` uses `logp_descriptor_selection_results.tsv`. For other filenames, the train table stem is the prefix. The old unprefixed rank file is not reused. Remove a split's rank file if its training data changes.
