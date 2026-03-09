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


def _concat_batches(items: List[torch.Tensor]) -> torch.Tensor:
    if not items:
        raise ValueError("No tensors to concatenate.")
    return torch.cat(items, dim=0)


def run_evaluation(config: Dict[str, Any], runtime: Dict[str, Any]) -> EvalResult:
    model, data, metric_names, metrics, task_names, tasks = _instantiate_plugins(config, runtime)
    model.setup()
    dataloader = data.build_dataloader()

    true_batches: List[torch.Tensor] = []
    pred_batches: List[torch.Tensor] = []
    mask_batches: List[torch.Tensor] = []
    labels_store: Dict[str, List[torch.Tensor]] = defaultdict(list)
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
        else:
            default_mask = torch.ones_like(batch.ecg, dtype=torch.bool).cpu()
            mask_batches.append(default_mask)

        if batch.labels:
            for key, value in batch.labels.items():
                labels_store[key].append(value.detach().cpu())

        seen_samples += int(batch.ecg.shape[0])

    ecg_true = _concat_batches(true_batches)
    ecg_pred = _concat_batches(pred_batches)
    mask = _concat_batches(mask_batches)

    labels = None
    if labels_store:
        labels = {k: _concat_batches(v) for k, v in labels_store.items()}

    metric_results: Dict[str, float] = {}
    for name, metric_obj in zip(metric_names, metrics):
        out = metric_obj.compute(ecg_true=ecg_true, ecg_pred=ecg_pred, mask=mask, meta={})
        for key, value in out.items():
            metric_results[f"{name}.{key}"] = float(value)

    task_results: Dict[str, Dict[str, float]] = {}
    for name, task_obj in zip(task_names, tasks):
        out = task_obj.evaluate(ecg_true=ecg_true, ecg_pred=ecg_pred, labels=labels, mask=mask, meta={})
        task_results[name] = {k: float(v) for k, v in out.items()}

    summary = {
        "num_samples": seen_samples,
        "num_timesteps": int(mask.sum().item()),
        "model": config["model"]["name"],
        "data": config["data"]["name"],
    }
    return EvalResult(metrics=metric_results, tasks=task_results, summary=summary)
