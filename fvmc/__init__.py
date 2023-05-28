# import scipy.signal as _signal
# del _signal # just to avoid import failure of jax

from jax.config import config as _jax_config
_jax_config.update("jax_enable_x64", True)

del _jax_config

import logging
logging.basicConfig(force=True, format='# [%(asctime)s] %(levelname)s: %(message)s')
LOGGER = logging.getLogger("fvmc")
del logging

# import warnings
# with warnings.catch_warnings():
#     warnings.simplefilter("ignore", category=DeprecationWarning)
#     from . import (
#         utils,
#         hamiltonian,
#         wavefunction,
#         sampler,
#         estimator,
#         optimizer,
#         preconditioner,
#         config,
#         train,
#         neuralnet,
#     )

__all__ = [
    "utils",
    "hamiltonian",
    "wavefunction",
    "sampler",
    "estimator",
    "optimizer",
    "preconditioner",
    "config",
    "train",
    "neuralnet",
]

def __getattr__(name):
    from importlib import import_module
    import warnings
    if name in __all__:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=DeprecationWarning)
            return import_module("." + name, __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")