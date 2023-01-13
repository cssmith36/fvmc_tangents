# import scipy.signal as _signal
# del _signal # just to avoid import failure of jax

from jax.config import config as _jax_config
_jax_config.update("jax_enable_x64", True)

del _jax_config

from . import (
    utils,
    hamiltonian,
    wavefunction,
    sampler,
    estimator,
    optimizer,
)

# __all__ = [
#     "utils",
#     "hamiltonian",
#     "wavefunction",
#     "sampler",
#     "estimator",
#     "optimizer",
# ]

# def __getattr__(name):
#     from importlib import import_module
#     if name in __all__:
#         return import_module("." + name, __name__)
#     raise AttributeError(f"module {__name__!r} has no attribute {name!r}")