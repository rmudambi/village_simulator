import dataclasses
from typing import Callable, Dict, Union

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
    params: Dict[str, float]

    def sample(
        self, randomness_stream: RandomnessStream, additional_key: str, index: pd.Index = None
    ) -> Union[float, pd.Series]:
        # fixme: this would be much simpler if RandomnessStream could return a scalar
        if index is None:
            quantiles = randomness_stream.get_draw(pd.RangeIndex(1), additional_key)
        else:
            quantiles = randomness_stream.get_draw(index, additional_key)

        values = self.distribution.ppf(quantiles, **self.params)

        if index is None:
            values = values[0]
        else:
            values = pd.Series(values, index=index, name=additional_key)
        return values


def from_configuration(
    configuration: ConfigTree,
    randomness_stream: RandomnessStream,
    additional_key: str,
    index: pd.Index = None,
) -> Union[float, pd.Series]:
    distribution_parameters = {
        key: from_configuration(
            configuration[key], randomness_stream, f"{additional_key}_{key}"
        )
        if isinstance(configuration[key], ConfigTree)
        else configuration[key]
        for key in configuration
        if key != "distribution"
    }
    distribution = FrozenDistribution(configuration.distribution, distribution_parameters)
    return distribution.sample(randomness_stream, additional_key, index)
