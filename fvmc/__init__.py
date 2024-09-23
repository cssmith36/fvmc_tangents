from jax import config as _jax_config
_jax_config.update("jax_enable_x64", True)

del _jax_config

import logging
logging.basicConfig(force=True, format='# [%(asctime)s] %(levelname)s: %(message)s')
LOGGER = logging.getLogger("fvmc")
del logging


__all__ = [
    "utils",
    "moire",
    "ewaldsum",
    "hamiltonian",
    "sampler",
    "estimator",
    "optimizer",
    "preconditioner",
    "wavefunction",
    "config",
    "train",
    "observable",
    "tangents",
]

def __getattr__(name):
    from importlib import import_module
    if name in __all__:
        return import_module("." + name, __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
