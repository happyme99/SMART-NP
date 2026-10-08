# Data layout

Large inputs stay outside Git. Download them from the data-v1 release as shown in the [root README](../README.md), then pass each required file explicitly.

Pass explicit train/test/calibration files to `scripts/run.sh`:

```text
./scripts/run.sh \
  --train-table data/dataset/random_train.tsv --train-embeddings data/dataset/gin_random_train_emb.pt \
  --test-table data/dataset/random_test.tsv --test-embeddings data/dataset/gin_random_test_emb.pt --test-softmax data/dataset/gin_random_test_softmax.pt \
  --calibration-table data/dataset/gin_random_calibration.tsv --calibration-softmax data/dataset/gin_random_calibration_softmax.pt
```

First run automatically writes `data/descriptor-ranks/descriptor_selection_results.tsv`, then reuses it.
