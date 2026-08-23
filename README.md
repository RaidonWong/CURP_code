# CURP

CURP converts a user's profile history into quantized prototype representations,
extracts stable and query-dependent preferences, and injects the resulting soft
tokens into a frozen causal language model.

This repository is a portable code release. It intentionally excludes datasets,
embeddings, checkpoints, generated predictions, logs, cached dependencies, and
machine-specific configuration.

## Method

```text
profile history
  -> text encoder embeddings
  -> PQ / RQ / OPQ reconstruction
  -> stable Q-Former + dynamic cross-attention
  -> MLP projection
  -> soft-token injection into a frozen LLM
```

The OPQ option implemented here uses a covariance-eigenvector rotation with
variance-aware subspace allocation, followed by per-subspace PQ. It is a
lightweight rotated-PQ variant rather than the iterative rotation/codebook
optimization used by classical OPQ implementations.

## Included code

- `prepare_data.py`: convert LaMP-style inputs and outputs to CURP JSONL.
- `encode.py`, `encode_parallel.py`: encode profile items.
- `stage1.py`: train the default PQ codebook.
- `build_altquant_codebook.py`: build RQ or rotated-PQ/OPQ codebooks.
- `stage2.py`: train the stable and dynamic personalization modules.
- `infer.py`: run generation from a Stage-2 checkpoint.
- `evaluate.py`: classification, rating, and generation metrics.
- `stage2_*.py`, `infer_*.py`: paper ablations and encoder variants.
- `tests/`: CPU regression tests.

## Installation

Create a Python environment, install a PyTorch build suitable for the target
machine, and then install the remaining dependencies:

```bash
python -m pip install -r requirements.txt
```

Optional generation metrics use `bert-score`, `nltk`, and `rouge-score`.

## Configuration

No local model or filesystem address is stored in the source code. Configure
resources at runtime:

```bash
export LLM_PATH="<model directory or model identifier>"
export ENCODER_PATH="<encoder directory or model identifier>"
```

The pipeline also accepts these optional environment variables:

- `DATA_DIR`
- `STAGE1_DIR`
- `STAGE2_DIR`
- `RESULTS_DIR`
- `BERT_MODEL_PATH`
- `PYTHON_BIN`
- `ACCELERATE_BIN`

Blank variable names are listed in `config.example.env`. Do not commit a filled
configuration file containing private locations or credentials.

## Expected data layout

The default pipeline expects:

```text
data/
  lamp2_train.jsonl
  lamp2_train_embeddings.pt
  lamp2_val.jsonl
  lamp2_val_embeddings.pt
```

Each processed sample has the following structure:

```json
{
  "id": "sample-id",
  "input": "current query",
  "output": "reference answer",
  "profile": [{"order": 17, "text": "profile item"}]
}
```

`order` indexes the first dimension of the corresponding embedding tensor.

## Run one task

Default PQ:

```bash
bash run_task.sh lamp2 0
```

Rotated-PQ/OPQ:

```bash
QUANTIZER=opq bash run_task.sh lamp2 0
```

Residual quantization:

```bash
QUANTIZER=rq bash run_task.sh lamp2 0
```

Set `SKIP_STAGE1=1` to reuse an existing codebook. Hyperparameters can be
overridden with the environment variables used in `run_task.sh`.

## Run multiple tasks

```bash
TASK_LIST="lamp2 lamp4" bash run_all.sh 0
```

## Tests

```bash
python -m unittest discover -s tests -v
```

The tests cover tokenization, placeholder injection, PQ/RQ/OPQ tensor shapes,
portable checkpoint metadata, Stage-1 edge cases, and evaluation fallbacks.

## Privacy and release policy

- Checkpoint metadata stores resource names, not absolute resource paths.
- Runtime paths are supplied only through command-line arguments or environment
  variables.
- `.gitignore` excludes data, weights, predictions, logs, caches, and local
  configuration.
- Before publishing new files, run `python tools/check_release.py`.
