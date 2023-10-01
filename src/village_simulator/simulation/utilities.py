from typing import Union

import numpy as np
import pandas as pd
from scipy import stats
from vivarium import ConfigTree
from vivarium.framework.randomness import RandomnessStream


def _round_series_stochastic(
    values: pd.Series, randomness_stream: RandomnessStream, additional_key: str = ""
) -> pd.Series:
    rounding_propensities = randomness_stream.get_draw(
        values.index, additional_key=f"{additional_key}_stochastic_rounding"
    )
    floored_values = np.floor(values)
    thresholds = values - floored_values
    values = (rounding_propensities < thresholds) + floored_values
    return values.astype(int)


def round_stochastic(
    values: Union[pd.DataFrame, pd.Series],
    randomness_stream: RandomnessStream,
    additional_key: str = "",
) -> Union[pd.DataFrame, pd.Series]:
    if isinstance(values, pd.DataFrame):
        return values.apply(
            lambda column: _round_series_stochastic(
                column, randomness_stream, f"{column.name}_{additional_key}"
            )
        )
    else:
        return _round_series_stochastic(values, randomness_stream, additional_key)


def sample_from_normal_distribution(
    index: pd.Index,
    configuration: ConfigTree,
    randomness_stream: RandomnessStream,
    additional_key: str = "",
) -> pd.Series:
    values = stats.norm.ppf(
        randomness_stream.get_draw(index, additional_key),
        loc=configuration.mean,
        scale=configuration.standard_deviation,
    )
    return pd.Series(values)