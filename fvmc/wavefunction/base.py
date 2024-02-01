import abc
import functools
import dataclasses
from typing import Any, Callable, Sequence, Tuple

import jax
from flax import linen as nn
from jax import numpy as jnp

from ..utils import ElecConf, NuclConf, Array


def model_wrapper(model: nn.Module,
                  wrap_out: Callable[[Array, Array], Any]):
    # switch between different model type
    if isinstance(model, FullWfn): # combine r, x into a tuple conf = (r, x)
        return lambda p, conf: wrap_out(*model.apply(p, *conf))
    elif isinstance(model, ElecWfn): # electron only case
        return lambda p, x: wrap_out(*model.apply(p, x))
    else: # fall back for any model
        # raise TypeError(f"unsupoorted model type {type(model)}")
        return lambda p, *a, **kw: wrap_out(*model.apply(p, *a, **kw))

def _wrap_sign(sign, logp):
    if jnp.iscomplexobj(sign):
        logp += jnp.log(sign)
    return logp

log_prob_from_model = functools.partial(model_wrapper, wrap_out=lambda s, l: 2*l)
log_psi_from_model = functools.partial(model_wrapper, wrap_out=_wrap_sign)


class FullWfn(nn.Module, abc.ABC):
    @abc.abstractmethod
    def __call__(self, r: Array, x: Array) -> Tuple[Array, Array]:
        """Take nuclei position r and electron position x, return sign and log|psi|"""
        raise NotImplementedError


class ElecWfn(nn.Module, abc.ABC):
    @abc.abstractmethod
    def __call__(self, x: Array) -> Tuple[Array, Array]:
        """Take only the electron position x, return sign and log|psi|"""
        raise NotImplementedError


@dataclasses.dataclass
class FakeModel:
    fn: Callable
    init_params: Any

    def init(self, *args, **kwargs):
        return self.init_params

    def apply(self, params, *args):
        return self.fn(params, *args)


class FixNuclei(ElecWfn):
    r"""Module warpper that fix the nuclei positions for a full model

    This class takes a full wavefunction model f(r,x) of r (nuclei) and x (electrons)
    and the fixed nuclei positions r_0, and return a new model which only depends on x.
    Think it as a partial warpper that works on nn.Module
    """
    model: FullWfn
    nuclei: NuclConf

    def __call__(self, x: ElecConf) -> Tuple[Array, Array]:
        return self.model(r=self.nuclei, x=x)


class ProductModel(FullWfn):
    r"""Pruduct of multiple model results.

    Assuming the models returns in log scale.
    The signature of each submodel can either be pure: x -> log(f(x))
    or with sign: x -> sign(f(x)), log(|f(x)|).
    The model will return sign if any of its submodels returns sign.
    """

    submodels: Sequence[nn.Module]

    @nn.compact
    def __call__(self, r:NuclConf, x: ElecConf) -> Tuple[Array, Array]:
        sign = 1.
        logf = 0.
        with_sign = True # False will make the sign optional

        for model in self.submodels:
            result = model(r, x)
            if isinstance(result, tuple):
                sign *= result[0]
                logf += result[1]
                with_sign = True
            else:
                logf += result

        if with_sign:
            return sign, logf
        else:
            return logf
