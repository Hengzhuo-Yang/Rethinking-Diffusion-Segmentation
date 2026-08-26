"""Runtime compatibility shims for legacy SDSeg dependencies on new Python images."""

import collections
import collections.abc
import warnings

for _name in ("Callable", "Mapping", "MutableMapping", "Sequence", "Iterable", "MutableSet"):
    if not hasattr(collections, _name) and hasattr(collections.abc, _name):
        setattr(collections, _name, getattr(collections.abc, _name))

try:
    import numpy as _np

    for _name, _value in {
        "bool": bool,
        "int": int,
        "float": float,
        "complex": complex,
        "object": object,
    }.items():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            _missing = not hasattr(_np, _name)
        if _missing:
            setattr(_np, _name, _value)
except Exception:
    pass
