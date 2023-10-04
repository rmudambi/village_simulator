from typing import Callable, Dict

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


_DISTRIBUTIONS: Dict[str, Callable[[pd.Series, ...], pd.Series]] = {
    "normal": stats.norm.ppf,
    "zero_inflated_gamma": _zero_inflated_gamma_ppf,
}


def sample_from_distribution(
    configuration: ConfigTree,
    randomness_stream: RandomnessStream,
    additional_key: str,
    index: pd.Index,
) -> pd.Series:
    params = {key: configuration[key] for key in configuration if key != "distribution"}
    values = _DISTRIBUTIONS[configuration.distribution](
        randomness_stream.get_draw(index, additional_key), **params
    )
    return pd.Series(values, name=additional_key)
