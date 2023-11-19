import numpy as np
import pandas as pd
from scipy import stats

from village_simulator.simulation.components.weather import (
    DRY_PROBABILITY,
    GAMMA_SCALE_PARAMETER,
    GAMMA_SHAPE_PARAMETER,
    RAINFALL_LOCAL_VARIABILITY,
    RAINFALL_SEASONALITY_MAX,
    RAINFALL_SEASONALITY_MIN,
    RAINFALL_SEASONALITY_MIN_DATE,
)
from village_simulator.simulation.distributions import stretched_truncnorm_ppf
from village_simulator.simulation.utilities import get_value_from_annual_cycle


def sample_rainfall_in_period(
    period: pd.DatetimeIndex, num_years: int, num_tiles: int
) -> pd.Series:
    """
    Generate samples of rainfall accumulation during the input annualized period
    for a specified number of years and local tiles.
    """

    # get aridity factor for each day in period
    aridity_factors = {
        date: get_value_from_annual_cycle(
            time=date,
            min=RAINFALL_SEASONALITY_MIN,
            max=RAINFALL_SEASONALITY_MAX,
            min_date=pd.Timestamp(
                2000,
                RAINFALL_SEASONALITY_MIN_DATE["month"],
                RAINFALL_SEASONALITY_MIN_DATE["day"],
            ),
        )
        for date in period
    }
    aridity_factors = pd.Series(
        aridity_factors.values(), index=pd.Index(aridity_factors.keys(), name="date")
    )
    aridity_factors = pd.DataFrame(
        np.broadcast_to(aridity_factors.values[:, None], (len(aridity_factors), num_years)),
        index=aridity_factors.index,
        columns=pd.Index(range(num_years), name="year"),
    ).stack()
    aridity_factors_values = aridity_factors.values[:, None]

    # get zero_inflated_gamma_ppf parameters for each day in period
    dry_probabilities = 1 - aridity_factors * (1 - DRY_PROBABILITY)
    scale = aridity_factors * GAMMA_SCALE_PARAMETER

    # sample large number of observations for each day in period
    is_dry_propensity = pd.Series(
        np.random.rand(len(aridity_factors)), index=aridity_factors.index
    )
    not_dry_index = dry_probabilities[dry_probabilities >= is_dry_propensity].index

    regional_rainfall = pd.Series(0.0, index=aridity_factors.index)
    regional_rainfall.loc[not_dry_index] = stats.gamma.ppf(
        pd.Series(np.random.rand(len(not_dry_index)), index=not_dry_index),
        a=GAMMA_SHAPE_PARAMETER,
        scale=scale[not_dry_index],
    )

    expected_rainfall = pd.DataFrame(
        np.broadcast_to(
            regional_rainfall.values[:, None], (len(regional_rainfall), num_tiles)
        ),
        index=regional_rainfall.index,
        columns=pd.Index(range(num_tiles), name="tile"),
    ).stack()

    # use stretched truncnorm to get local values for large number of locations
    local_rainfall = stretched_truncnorm_ppf(
        pd.Series(np.random.rand(len(expected_rainfall)), index=expected_rainfall.index),
        loc=expected_rainfall.values[:, None],
        scale=RAINFALL_LOCAL_VARIABILITY,
    )

    return local_rainfall
