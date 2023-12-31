from typing import Dict, List

import numpy as np
import pandas as pd
from scipy import stats
from vivarium import Component
from vivarium.framework.engine import Builder
from vivarium.framework.event import Event
from vivarium.framework.population import SimulantData

from village_simulator.constants import Columns, Pipelines
from village_simulator.simulation.distributions import stretched_truncnorm_ppf
from village_simulator.simulation.utilities import get_value_from_annual_cycle

RAINFALL_SEASONALITY_MIN = 0.1
RAINFALL_SEASONALITY_MAX = 1.0
RAINFALL_SEASONALITY_MIN_DATE = {"month": 8, "day": 15}

DRY_PROBABILITY = 0.55
GAMMA_SHAPE_PARAMETER = 0.9902
GAMMA_SCALE_PARAMETER = 15.0
RAINFALL_LOCAL_VARIABILITY = 0.05


class Weather(Component):
    """Component that manages the weather"""

    CONFIGURATION_DEFAULTS = {
        "weather": {
            Pipelines.TEMPERATURE: {
                "mean": 65.0,
                "seasonality": {
                    "amplitude": 15.0,
                    "min_date": {"month": 1, "day": 15},
                },
                "stochastic_variability": 5.0,
                "local_variability": 2.0,
            },
            Pipelines.RAINFALL: {
                "seasonality": {
                    "min": RAINFALL_SEASONALITY_MIN,
                    "max": RAINFALL_SEASONALITY_MAX,
                    "min_date": RAINFALL_SEASONALITY_MIN_DATE,
                },
                "dry_probability": DRY_PROBABILITY,
                "gamma_shape_parameter": GAMMA_SHAPE_PARAMETER,
                "gamma_scale_parameter": GAMMA_SCALE_PARAMETER,
                "local_variability": RAINFALL_LOCAL_VARIABILITY,
            },
        }
    }

    ##############
    # Properties #
    ##############

    @property
    def columns_created(self) -> List[str]:
        return [Columns.TEMPERATURE, Columns.RAINFALL]

    @property
    def initialization_requirements(self) -> Dict[str, List[str]]:
        return {"requires_streams": [self.name]}

    @property
    def time_step_prepare_priority(self) -> int:
        return 2

    #####################
    # Lifecycle methods #
    #####################

    def setup(self, builder: Builder) -> None:
        self.configuration = builder.configuration.weather
        self.randomness = builder.randomness.get_stream(self.name)

        self.get_temperature = builder.value.register_value_producer(
            Pipelines.TEMPERATURE, self.temperature_source
        )

        self.get_rainfall = builder.value.register_value_producer(
            Pipelines.RAINFALL, self.rainfall_source
        )

    def on_initialize_simulants(self, pop_data: SimulantData) -> None:
        self.population_view.update(
            pd.DataFrame(
                {column: np.nan for column in self.columns_created}, index=pop_data.index
            )
        )

    def on_time_step_prepare(self, event: Event) -> None:
        temperature = self.get_temperature(event)
        rainfall = self.get_rainfall(event)
        self.population_view.update(pd.concat([temperature, rainfall], axis=1))

    ####################
    # Pipeline sources #
    ####################

    def temperature_source(self, event: Event) -> pd.Series:
        config = self.configuration.temperature

        seasonal_temperature_shift = get_value_from_annual_cycle(
            event.time,
            amplitude=config.seasonality.amplitude,
            min_date=config.seasonality.min_date,
        )
        expected_temperature = config.mean + seasonal_temperature_shift
        regional_temperature = self.randomness.sample_from_distribution(
            pd.Index([0]),
            distribution=stats.norm,
            additional_key="regional_temperature",
            loc=expected_temperature,
            scale=config.stochastic_variability,
        )[0]

        temperatures = self.randomness.sample_from_distribution(
            event.index,
            distribution=stats.norm,
            additional_key="temperature",
            loc=regional_temperature,
            scale=config.local_variability,
        )
        return temperatures

    def rainfall_source(self, event: Event) -> pd.Series:
        config = self.configuration.rainfall

        aridity_factor = get_value_from_annual_cycle(
            event.time,
            min=config.seasonality.min,
            max=config.seasonality.max,
            min_date=config.seasonality.min_date,
        )
        dry_probability = 1 - aridity_factor * (1 - config.dry_probability)
        is_dry_propensity = self.randomness.get_draw(
            pd.Index([0]),
            additional_key="is_dry",
        )[0]

        if is_dry_propensity < dry_probability:
            return pd.Series(0.0, index=event.index)

        scale = aridity_factor * config.gamma_scale_parameter

        regional_rainfall = self.randomness.sample_from_distribution(
            pd.Index([0]),
            distribution=stats.gamma,
            additional_key="regional_rainfall",
            a=config.gamma_shape_parameter,
            scale=scale,
        )[0]

        rainfall = self.randomness.sample_from_distribution(
            event.index,
            ppf=stretched_truncnorm_ppf,
            additional_key="rainfall",
            loc=regional_rainfall,
            scale=config.local_variability,
        )
        return rainfall
