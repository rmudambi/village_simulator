import numpy as np
import pandas as pd

from village_simulator.simulation.components.resources import WHEAT_HARVEST_DATE
from village_simulator.simulation.components.weather import (
    DRY_PROBABILITY,
    GAMMA_SCALE_PARAMETER,
    GAMMA_SHAPE_PARAMETER,
    RAINFALL_LOCAL_VARIABILITY,
    RAINFALL_SEASONALITY_MAX,
    RAINFALL_SEASONALITY_MIN,
    RAINFALL_SEASONALITY_MIN_DATE,
)
from village_simulator.simulation.distributions import (
    stretched_truncnorm_ppf,
    zero_inflated_gamma_ppf,
)
from village_simulator.simulation.utilities import get_value_from_annual_cycle


def sample_mid_growth_rainfall(years: int, tiles: int) -> float:
    # get mid-growth period
    harvest_date = pd.Timestamp(2000, WHEAT_HARVEST_DATE["month"], WHEAT_HARVEST_DATE["day"])
    start_date = harvest_date - pd.Timedelta(days=180)
    end_date = start_date + pd.Timedelta(days=90)
    date_range = pd.date_range(start_date, end_date)

    # get aridity factor for each day in period
    aridity_factors = [
        get_value_from_annual_cycle(
            time=date,
            min=RAINFALL_SEASONALITY_MIN,
            max=RAINFALL_SEASONALITY_MAX,
            min_date=pd.Timestamp(
                2000,
                RAINFALL_SEASONALITY_MIN_DATE["month"],
                RAINFALL_SEASONALITY_MIN_DATE["day"],
            ),
        )
        for date in date_range
    ]
    aridity_factors = np.tile(aridity_factors, years)[:, None]

    # get zero_inflated_gamma_ppf parameters for each day in period
    dry_probabilities = 1 - aridity_factors * (1 - DRY_PROBABILITY)
    scale = aridity_factors * GAMMA_SCALE_PARAMETER

    # sample large number of observations for each day in period
    regional_rainfall = zero_inflated_gamma_ppf(
        pd.Series(np.random.rand(len(aridity_factors))),
        p_zero=dry_probabilities,
        shape=GAMMA_SHAPE_PARAMETER,
        scale=scale,
    )

    expected_rainfall = np.tile(regional_rainfall.values, tiles)[:, None]
    # use stretched truncnorm to get local values for large number of locations
    local_rainfall = stretched_truncnorm_ppf(
        pd.Series(np.random.rand(len(expected_rainfall))),
        loc=expected_rainfall,
        scale=RAINFALL_LOCAL_VARIABILITY,
    )

    # get mean mid-growth rainfall
    mean_all_years_rainfall = (local_rainfall / tiles).sum()
    mean_rainfall = mean_all_years_rainfall / years
    return mean_rainfall
