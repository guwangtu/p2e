# Evaluation Module (Plugin Architecture)

This folder provides a unified evaluation framework for PPG-to-ECG pipelines.

## Goals

- Support multiple model backends through model adapters.
- Support variable-length dataloaders through data adapters and masks.
- Support both generic metrics and dataset-specific downstream task heads.

## Directory Layout

- `run_eval.py`: CLI entrypoint.
- `configs/`: YAML configs for experiments.
- `core/`: registry, shared types, and evaluation runner.
- `adapters/models/`: pluggable model adapters.
- `adapters/data/`: pluggable dataset adapters.
- `metrics/`: generic signal-quality metrics.
- `tasks/`: downstream task heads (BP, stress, etc.).
- `outputs/`: default output folder.

## Quick Start

```bash
python p2e/evaluation/run_eval.py --config p2e/evaluation/configs/base.yaml
```

### UNet1D Smoke Test

```bash
python p2e/evaluation/run_eval.py --config p2e/evaluation/configs/unet1d_smoke.yaml --device cpu
```

`unet1d` adapter loads `p2e/scripts/models/unet1d.py::UNet1D`.
If you have a trained checkpoint, set `model.ckpt_path` in the config.

## How To Extend

1. Add a new adapter/head class implementing the corresponding base interface.
2. Register it in `run_eval.py` via `register_plugins(...)`.
3. Reference it in config:

```yaml
model:
  name: your_model_adapter
data:
  name: your_data_adapter
evaluation:
  metrics: [rmse]
  tasks: [your_task_head]
```
