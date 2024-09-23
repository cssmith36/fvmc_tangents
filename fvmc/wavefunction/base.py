import abc
import functools
import dataclasses
from typing import Any, Callable, Sequence, Tuple

import jax
import flax
from flax import linen as nn
from jax import numpy as jnp
from jax import tree_util as jtu

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

def frozen_model_wrapper(model: nn.Module,
                  wrap_out: Callable[[Array, Array], Any]):
    # switch between different model type
    if isinstance(model, FullWfn): # combine r, x into a tuple conf = (r, x)
        return lambda p, p2, conf: wrap_out(*model.apply(p, p2, *conf))
    elif isinstance(model, ElecWfn): # electron only case
        return lambda p, p2, x: wrap_out(*model.apply(p, p2, x))
    else: # fall back for any model
        # raise TypeError(f"unsupoorted model type {type(model)}")
        return lambda p, p2, *a, **kw: wrap_out(*model.apply(p, p2, *a, **kw))

def _wrap_sign(sign, logp):
    if jnp.iscomplexobj(sign):
        logp += jnp.log(sign)
    return logp

log_prob_from_model = functools.partial(model_wrapper, wrap_out=lambda s, l: 2*l)
log_psi_from_model = functools.partial(model_wrapper, wrap_out=_wrap_sign)
log_psi_from_frozen_model = functools.partial(frozen_model_wrapper, wrap_out=_wrap_sign)


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


@dataclasses.dataclass
class FakeModel:
    fn: Callable
    init_params: Any

    def init(self, *args, **kwargs):
        return self.init_params

    def apply(self, params, *args, **kwargs):
        return self.fn(params, *args, **kwargs)


@dataclasses.dataclass
class FrozenModel:
    model: nn.Module
    frozen_params: Any

    def init(self, rng, *args, **kwargs):
        raw_params = self.model.init(rng, *args, **kwargs)
        return jtu.tree_map(
            lambda x, y: y if x is None else None,
            self.frozen_params, raw_params, is_leaf=_is_none)

    def apply(self, params, *args, **kwargs):
        all_params = jtu.tree_map(
            lambda x, y: y if x is None else x,
            self.frozen_params, params, is_leaf=_is_none)
        return self.model.apply(all_params, *args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        if hasattr(nn.Module, name):
            raise AttributeError(
                "only `init` and `apply` are allowed in FrozenModel")
        return getattr(self.model, name)

@dataclasses.dataclass
class FrozenModelD2:
    model: nn.Module
    frozen_params: Any

    def init(self, rng, *args, **kwargs):
        raw_params = self.model.init(rng, *args, **kwargs)
        return jtu.tree_map(
            lambda x, y: y if x is None else None,
            self.frozen_params, raw_params, is_leaf=_is_none)

    def apply(self, params,backflow_params, *args, **kwargs):
        if backflow_params:
            params['params'][str(self.frozen_params)] = backflow_params
        #all_params = jtu.tree_map(
        #    lambda x, y: y if x is None else x,
        #    self.frozen_params, params, is_leaf=_is_none)
        return self.model.apply(params, *args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        if hasattr(nn.Module, name):
            raise AttributeError(
                "only `init` and `apply` are allowed in FrozenModel")
        return getattr(self.model, name)


@dataclasses.dataclass
class FrozenModelD3:
    model: nn.Module
    frozen_params: Any

    #def init(self, rng, *args, **kwargs):
    #    raw_params = self.model.init(rng, *args, **kwargs)
    #    return jtu.tree_map(
    #        lambda x, y: y if x is None else None,
    #        self.frozen_params, raw_params, is_leaf=_is_none)

    def apply(self, d2params, d1params, *args, **kwargs):
        
        params = flax.core.copy(d1params)
        params['params']['submodels_0']['PlanewaveOrbital_0']['Dense_0']['kernel'] = d1params['params']['submodels_0']['PlanewaveOrbital_0']['Dense_0']['kernel'].at[21-4:21+4,:].set(d2params)

        #replaced_params = params.at[21-5:21+5,:].set(d2params)
        #d1params['params']['submodels_0']['PlanewaveOrbital_0']['Dense_0']['kernel'] = replaced_params
        

        return self.model.apply(params, *args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        if hasattr(nn.Module, name):
            raise AttributeError(
                "only `init` and `apply` are allowed in FrozenModel")
        return getattr(self.model, name)

_is_none = lambda x: x is None
