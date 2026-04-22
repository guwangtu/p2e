from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise ImportError("PyYAML is required. Install with `pip install pyyaml`.") from exc

# Ensure `p2e/evaluation` is importable as top-level modules (core/adapters/metrics/tasks).
EVAL_ROOT = Path(__file__).resolve().parent
if str(EVAL_ROOT) not in sys.path:
    sys.path.insert(0, str(EVAL_ROOT))

from core.runner import run_evaluation

# Import plugins so decorators register them.
from adapters.models import dummy_adapter  # noqa: F401
from adapters.models import rddm_adapter  # noqa: F401
from adapters.models import unet1d_adapter  # noqa: F401
from adapters.models import cycle_flow_adapter  # noqa: F401
from adapters.models import cardio_wm_adapter  # noqa: F401
from adapters.data import bidmc_mock_adapter  # noqa: F401
from adapters.data import mimic_adapter  # noqa: F401
from metrics import fd_placeholder  # noqa: F401
from metrics import rmse  # noqa: F401
from metrics import mae  # noqa: F401
from metrics import pearson  # noqa: F401
from metrics import dtw_metric  # noqa: F401
from metrics import rpeak_error  # noqa: F401
from metrics import nfe  # noqa: F401
from tasks import bp_estimation  # noqa: F401
from tasks import stress_fatigue  # noqa: F401


def _load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _save_result(save_dir: str, payload: Dict[str, Any]) -> str:
    os.makedirs(save_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(save_dir, f"result_{ts}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=str, help="Path to YAML config")
    parser.add_argument("--device", default="cpu", type=str, help="Runtime device")
    parser.add_argument("--seed", default=42, type=int, help="Runtime seed")
    parser.add_argument("--save_dir", default="", type=str, help="Override output dir")
    args = parser.parse_args()

    config = _load_config(args.config)
    runtime = {"device": args.device, "seed": args.seed}
    result = run_evaluation(config=config, runtime=runtime)

    save_dir = args.save_dir or config.get("evaluation", {}).get("save_dir", "p2e/evaluation/outputs")
    payload = {
        "config": config,
        "runtime": runtime,
        "summary": result.summary,
        "metrics": result.metrics,
        "tasks": result.tasks,
    }
    out_file = _save_result(save_dir=save_dir, payload=payload)

    print("\n=== Evaluation Summary ===")
    print(json.dumps(result.summary, indent=2, ensure_ascii=False))
    print("\n=== Metrics ===")
    print(json.dumps(result.metrics, indent=2, ensure_ascii=False))
    print("\n=== Tasks ===")
    print(json.dumps(result.tasks, indent=2, ensure_ascii=False))
    print(f"\nSaved result: {out_file}")


if __name__ == "__main__":
    main()
