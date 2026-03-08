from datetime import datetime
from importlib import import_module
from pathlib import Path
from typing import Any


def make_run_dir(output_dir: str, exp_name: str) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(output_dir) / f"{ts}_{exp_name}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir

import inspect

def instantiate(target: str, params: dict | None = None, **extra: Any):
    """
    target: "package.module.ClassName" or "package.module:function"
    params: kwargs dict
    extra: additional kwargs merged into params (extra overrides params)
    """
    if not target or "." not in target:
        raise ValueError(f"Invalid target: {target}")

    module_path, name = target.rsplit(".", 1)
    mod = import_module(module_path)
    obj = getattr(mod, name)

    kwargs = dict(params or {})
    kwargs.update(extra)
    #return obj(**kwargs)
    sig = inspect.signature(obj)
    valid_keys = set(sig.parameters.keys())

    filtered_kwargs = {}
    dropped = {}

    for k, v in kwargs.items():
        if k in valid_keys:
            filtered_kwargs[k] = v
        else:
            dropped[k] = v

    if dropped:
        print(
            f"[instantiate] WARNING: dropped unexpected kwargs for {target}: "
            f"{list(dropped.keys())}"
        )

    return obj(**filtered_kwargs)
