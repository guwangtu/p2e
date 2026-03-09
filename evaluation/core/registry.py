from __future__ import annotations

from typing import Callable, Dict, Type


class PluginRegistry:
    def __init__(self) -> None:
        self.model_adapters: Dict[str, Type] = {}
        self.data_adapters: Dict[str, Type] = {}
        self.metrics: Dict[str, Type] = {}
        self.tasks: Dict[str, Type] = {}

    def _register(self, bucket: Dict[str, Type], name: str, cls: Type) -> None:
        key = name.strip().lower()
        if not key:
            raise ValueError("Plugin name cannot be empty.")
        if key in bucket:
            raise ValueError(f"Duplicate plugin registration: {key}")
        bucket[key] = cls

    def register_model(self, name: str) -> Callable:
        def decorator(cls: Type) -> Type:
            self._register(self.model_adapters, name, cls)
            return cls

        return decorator

    def register_data(self, name: str) -> Callable:
        def decorator(cls: Type) -> Type:
            self._register(self.data_adapters, name, cls)
            return cls

        return decorator

    def register_metric(self, name: str) -> Callable:
        def decorator(cls: Type) -> Type:
            self._register(self.metrics, name, cls)
            return cls

        return decorator

    def register_task(self, name: str) -> Callable:
        def decorator(cls: Type) -> Type:
            self._register(self.tasks, name, cls)
            return cls

        return decorator


REGISTRY = PluginRegistry()
