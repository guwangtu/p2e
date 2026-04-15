from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Tuple

import torch

from .registry import REGISTRY
from .types import EvalBatch, EvalResult


def _instantiate_plugins(config: Dict[str, Any], runtime: Dict[str, Any]):
    model_name = config["model"]["name"].lower()
    data_name = config["data"]["name"].lower()

    if model_name not in REGISTRY.model_adapters:
        raise KeyError(f"Unknown model adapter: {model_name}")
    if data_name not in REGISTRY.data_adapters:
        raise KeyError(f"Unknown data adapter: {data_name}")

    model = REGISTRY.model_adapters[model_name](config["model"], runtime)
    data = REGISTRY.data_adapters[data_name](config["data"], runtime)

    metric_names = [name.lower() for name in config.get("evaluation", {}).get("metrics", [])]
    task_names = [name.lower() for name in config.get("evaluation", {}).get("tasks", [])]

    metrics = []
    tasks = []
    for name in metric_names:
        if name not in REGISTRY.metrics:
            raise KeyError(f"Unknown metric plugin: {name}")
        metrics.append(REGISTRY.metrics[name](config.get("metric_params", {}).get(name, {})))
    for name in task_names:
        if name not in REGISTRY.tasks:
            raise KeyError(f"Unknown task plugin: {name}")
        tasks.append(REGISTRY.tasks[name](config.get("task_params", {}).get(name, {})))

    return model, data, metric_names, metrics, task_names, tasks


def _pad_and_concat(items: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pad variable-length tensors to the same length and concatenate, returning a mask."""
    if not items:
        raise ValueError("No tensors to concatenate.")

    # Check if all have the same last-dim size
    shapes = set(t.shape[1:] for t in items)
    if len(shapes) == 1:
        return torch.cat(items, dim=0), None  # no padding needed

    # Variable lengths: pad to max length in the last dimension
    max_len = max(t.shape[-1] for t in items)
    padded, masks = [], []
    for t in items:
        L = t.shape[-1]
        if L < max_len:
            pad_size = max_len - L
            t_padded = torch.nn.functional.pad(t, (0, pad_size))
            m = torch.zeros(*t.shape[:-1], max_len, dtype=torch.bool)
            m[..., :L] = True
        else:
            t_padded = t
            m = torch.ones(*t.shape[:-1], max_len, dtype=torch.bool)
        padded.append(t_padded)
        masks.append(m)
    return torch.cat(padded, dim=0), torch.cat(masks, dim=0)


def run_evaluation(config: Dict[str, Any], runtime: Dict[str, Any]) -> EvalResult:
    model, data, metric_names, metrics, task_names, tasks = _instantiate_plugins(config, runtime)
    model.setup()
    dataloader = data.build_dataloader()

    true_batches: List[torch.Tensor] = []
    pred_batches: List[torch.Tensor] = []
    mask_batches: List[torch.Tensor] = []
    labels_store: Dict[str, List[torch.Tensor]] = defaultdict(list)
    aux_store: Dict[str, Any] = {}
    seen_samples = 0

    for batch in dataloader:
        if not isinstance(batch, EvalBatch):
            raise TypeError("Data adapter must yield EvalBatch objects.")
        pred = model.predict(batch)
        if batch.ecg is None:
            raise ValueError("Batch.ecg is required for quality evaluation.")

        true_batches.append(batch.ecg.detach().cpu())
        pred_batches.append(pred.ecg_pred.detach().cpu())
        if batch.mask is not None:
            mask_batches.append(batch.mask.detach().cpu())

        if batch.labels:
            for key, value in batch.labels.items():
                if isinstance(value, torch.Tensor):
                    labels_store[key].append(value.detach().cpu())

        # Collect aux (e.g., nfe) from first batch
        if pred.aux and not aux_store:
            aux_store = pred.aux

        seen_samples += int(batch.ecg.shape[0])

    # Pad and concat (handles variable-length batches)
    ecg_true, pad_mask_true = _pad_and_concat(true_batches)
    ecg_pred, pad_mask_pred = _pad_and_concat(pred_batches)

    # Combine masks: user-provided mask AND padding mask
    if mask_batches:
        user_mask, _ = _pad_and_concat(mask_batches)
    else:
        user_mask = None

    if pad_mask_true is not None:
        mask = pad_mask_true
        if user_mask is not None:
            mask = mask & user_mask
    elif user_mask is not None:
        mask = user_mask
    else:
        mask = torch.ones_like(ecg_true, dtype=torch.bool)

    labels = None
    if labels_store:
        labels = {}
        for k, v in labels_store.items():
            try:
                labels[k] = torch.cat(v, dim=0)
            except RuntimeError:
                # Skip labels that can't be concatenated (different shapes)
                pass

    # Build meta dict for metrics (pass aux info like nfe)
    eval_meta = dict(aux_store)

    metric_results: Dict[str, float] = {}
    for name, metric_obj in zip(metric_names, metrics):
        out = metric_obj.compute(ecg_true=ecg_true, ecg_pred=ecg_pred, mask=mask, meta=eval_meta)
        for key, value in out.items():
            metric_results[f"{name}.{key}"] = float(value)

    task_results: Dict[str, Dict[str, float]] = {}
    for name, task_obj in zip(task_names, tasks):
        out = task_obj.evaluate(ecg_true=ecg_true, ecg_pred=ecg_pred, labels=labels, mask=mask, meta=eval_meta)
        task_results[name] = {k: float(v) for k, v in out.items()}

    summary = {
        "num_samples": seen_samples,
        "num_timesteps": int(mask.sum().item()),
        "model": config["model"]["name"],
        "data": config["data"]["name"],
    }
    return EvalResult(metrics=metric_results, tasks=task_results, summary=summary)
