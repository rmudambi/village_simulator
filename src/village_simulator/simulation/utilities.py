from typing import Union

import numpy as np
import pandas as pd
from vivarium import ConfigTree
from vivarium.framework.randomness import RandomnessStream
from vivarium.framework.time import Time

from village_simulator.simulation.constants import ONE_YEAR


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


def get_annual_time_stamp(year: int, time_config: ConfigTree):
    return pd.Timestamp(year, time_config["month"], time_config["day"])


def get_next_annual_event_date(
    clock_time: Time, event_month: int, event_day: int
) -> pd.Timestamp:
    """
    Return a timestamp corresponding to the next occurrence of an annual event.
    """
    if clock_time.month < event_month or (
        clock_time.month == event_month and clock_time.day < event_day
    ):
        event_year = clock_time.year
    else:
        event_year = clock_time.year + 1

    return pd.Timestamp(event_year, event_month, event_day)


def get_value_from_annual_cycle(config: ConfigTree, time: Time) -> float:
    min_day = get_annual_time_stamp(time.year, config.min_date)
    distance_from_minimum = (time - min_day) / ONE_YEAR

    mean = (config.max + config.min) * 0.5
    amplitude = 0.5 * (config.max - config.min)

    value = mean - amplitude * np.cos(2 * np.pi * distance_from_minimum)
    return value
