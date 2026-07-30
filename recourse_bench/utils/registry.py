from __future__ import annotations

import warnings

global_registry: dict[str, dict[str, type]] = {
    "Dataset": {},
    "PreProcess": {},
    "Method": {},
    "TargetModel": {},
    "Evaluation": {},
}

#: Retired registry names, per bucket, mapped to the name they were renamed to.
#: They keep existing configs working but warn on use, and are not listed by the
#: ``recourse_bench.list_*`` helpers.
deprecated_names: dict[str, dict[str, str]] = {
    "Dataset": {
        "toydata": "toy_data",
    },
    "TargetModel": {
        "randomforest": "random_forest",
    },
    "Evaluation": {
        "constraints": "constraint",
        "examples": "example",
    },
}


# Base class -> registry bucket, keyed by "<module basename>.<class name>".
# Matching on names rather than the classes themselves keeps this module free of
# imports: the component packages (``method/__init__.py`` and friends) import
# their components eagerly, so a base-object module can still be mid-import when
# a component elsewhere registers. Importing the bases here would make
# registration depend on which component happens to be imported first.
registry_bases: dict[str, str] = {
    "dataset_object.DatasetObject": "Dataset",
    "preprocess_object.PreProcessObject": "PreProcess",
    "method_object.MethodObject": "Method",
    "model_object.ModelObject": "TargetModel",
    "evaluation_object.EvaluationObject": "Evaluation",
}


def _resolve_registry_type(cls: type) -> str | None:
    """Map a class to its registry bucket by walking its MRO.

    Equivalent to an ``issubclass`` check against the five component base
    classes, but resolved by name so that no imports are needed — see
    :data:`registry_bases`.
    """
    for base in cls.__mro__:
        module = base.__module__.rsplit(".", 1)[-1]
        registry_type = registry_bases.get(f"{module}.{base.__qualname__}")
        if registry_type is not None:
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
        if name in deprecated_names.get(registry_type, {}):
            raise KeyError(
                f"{registry_type} '{name}' is a deprecated alias for "
                f"'{deprecated_names[registry_type][name]}' and cannot be reused"
            )

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
    return dict(global_registry[_resolve_registry_name(registry_type)])


def _resolve_registry_name(registry_type: str) -> str:
    mapping = {
        "dataset": "Dataset",
        "preprocess": "PreProcess",
        "method": "Method",
        "targetmodel": "TargetModel",  # Backup name
        "model": "TargetModel",
        "evaluation": "Evaluation",
    }
    type_name = mapping.get(registry_type.lower())
    if type_name is None:
        raise KeyError(f"Unknown registry type: {registry_type}")
    return type_name


def resolve_name(registry_type: str, name: str) -> str:
    """Map a component name to its current registry name.

    Names that were renamed keep working: passing a retired name returns the
    name it was renamed to and emits a :class:`DeprecationWarning`. Any other
    name is returned unchanged (including unknown ones — validation of the name
    itself is the caller's job).

    Parameters
    ----------
    registry_type : str
        One of ``"dataset"``, ``"preprocess"``, ``"method"``, ``"model"``
        (alias ``"targetmodel"``), or ``"evaluation"`` (case-insensitive).
    name : str
        Component name as written in a config.

    Returns
    -------
    str
        The current registry name.

    Raises
    ------
    KeyError
        If ``registry_type`` is unknown.
    """
    type_name = _resolve_registry_name(registry_type)
    current = deprecated_names.get(type_name, {}).get(name)
    if current is None:
        return name
    warnings.warn(
        f"{type_name} '{name}' was renamed to '{current}'; "
        f"update your config. The old name still works but will be removed.",
        DeprecationWarning,
        stacklevel=2,
    )
    return current
