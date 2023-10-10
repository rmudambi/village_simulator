import dataclasses
from typing import Callable, Dict, Optional, Union

import numpy as np
import pandas as pd
from scipy import stats
from vivarium import ConfigTree
from vivarium.framework.randomness import RandomnessStream


def _zero_inflated_gamma_ppf(
    quantiles: pd.Series, zero_probability: float, shape: float, scale: float
) -> pd.Series:
    values = pd.Series(0.0, index=quantiles.index)
    non_zero = quantiles > zero_probability
    quantiles[non_zero] = (quantiles[non_zero] - zero_probability) / (1 - zero_probability)
    values[non_zero] = stats.gamma.ppf(quantiles[non_zero], a=shape, scale=scale)
    return values


def _stretched_truncnorm_ppf(quantiles: pd.Series, loc: float, scale: float) -> np.ndarray:
    if loc == 0.0:
        return np.full(len(quantiles), 0.0)

    scale = scale * loc
    a = -loc / scale
    return stats.truncnorm.ppf(quantiles, a=a, b=5, loc=loc, scale=scale)


@dataclasses.dataclass
class Distribution:
    name: str
    ppf: Callable[[pd.Series, ...], Union[np.ndarray, pd.Series]]


NORMAL = Distribution("normal", stats.norm.ppf)
STRETCHED_TRUNCNORM = Distribution("stretched_truncnorm", _stretched_truncnorm_ppf)
ZERO_INFLATED_GAMMA = Distribution("zero_inflated_gamma", _zero_inflated_gamma_ppf)


@dataclasses.dataclass
class FrozenDistribution:
    distribution: Distribution
    params: Union[ConfigTree, Dict[str, Union[float, "FrozenDistribution"]]]

    def __post_init__(self):
        if isinstance(self.params, ConfigTree):
            self.params = self.params.to_dict()

    def sample(
        self,
        randomness_stream: RandomnessStream,
        additional_key: str,
        index: Optional[pd.Index],
    ) -> Union[float, pd.Series]:
        # fixme: this would be much simpler if RandomnessStream could return a scalar
        if index is None:
            quantiles = randomness_stream.get_draw(pd.RangeIndex(1), additional_key)
        else:
            quantiles = randomness_stream.get_draw(index, additional_key)

        params = {
            key: value.sample(randomness_stream, f"{additional_key}_{key}", None)
            if isinstance(value, FrozenDistribution)
            else value
            for key, value in self.params.items()
        }
        values = self.distribution.ppf(quantiles, **params)

        if index is None:
            values = values[0]
        else:
            values = pd.Series(values, index=index, name=additional_key)
        return values
