from __future__ import annotations

global_registry: dict[str, dict[str, type]] = {
    "Dataset": {},
    "PreProcess": {},
    "Method": {},
    "TargetModel": {},
    "Evaluation": {},
}


def _resolve_registry_type(cls: type) -> str | None:
    """Map a class to its registry bucket via ``issubclass``.

    Imported lazily so this module stays import-light and free of cycles with
    the base-object modules (which themselves never import the registry).
    """
    from recourse_bench.dataset.dataset_object import DatasetObject
    from recourse_bench.evaluation.evaluation_object import EvaluationObject
    from recourse_bench.method.method_object import MethodObject
    from recourse_bench.model.model_object import ModelObject
    from recourse_bench.preprocess.preprocess_object import PreProcessObject

    bases: tuple[tuple[type, str], ...] = (
        (DatasetObject, "Dataset"),
        (PreProcessObject, "PreProcess"),
        (MethodObject, "Method"),
        (ModelObject, "TargetModel"),
        (EvaluationObject, "Evaluation"),
    )
    for base, registry_type in bases:
        if issubclass(cls, base):
            return registry_type
    return None


def register(name: str):
    """Class decorator that registers a component under ``name``.

    The component's registry bucket (Dataset, PreProcess, Method, TargetModel,
    or Evaluation) is inferred from its base class. The same ``name`` may be
    reused across different buckets but must be unique within one.

    Parameters
    ----------
    name : str
        Registry key used in configs (e.g. ``dataset.name``, ``method.name``).

    Returns
    -------
    callable
        The class decorator.

    Raises
    ------
    TypeError
        If the class is not a recognized component subclass.
    KeyError
        If ``name`` is already registered in that bucket.

    Examples
    --------
    >>> @register("custom")
    ... class CustomDataset(DatasetObject):
    ...     ...
    """

    def decorator(cls: type) -> type:
        registry_type = _resolve_registry_type(cls)

        if registry_type is None:
            raise TypeError(f"Cannot register unsupported class type: {cls.__name__}")
        if name in global_registry[registry_type]:
            raise KeyError(f"{registry_type} '{name}' is already registered")

        global_registry[registry_type][name] = cls
        return cls

    return decorator


def get_registry(registry_type: str) -> dict[str, type]:
    """Return the registered classes for one component type.

    Parameters
    ----------
    registry_type : str
        One of ``"dataset"``, ``"preprocess"``, ``"method"``, ``"model"``
        (alias ``"targetmodel"``), or ``"evaluation"`` (case-insensitive).

    Returns
    -------
    dict[str, type]
        A shallow copy mapping registered names to component classes.

    Raises
    ------
    KeyError
        If ``registry_type`` is unknown.
    """
    registry_type = registry_type.lower()

    mapping = {
        "dataset": "Dataset",
        "preprocess": "PreProcess",
        "method": "Method",
        "targetmodel": "TargetModel",  # Backup name
        "model": "TargetModel",
        "evaluation": "Evaluation",
    }
    type_name = mapping.get(registry_type)
    if type_name is None:
        raise KeyError(f"Unknown registry type: {registry_type}")
    return dict(global_registry[type_name])
