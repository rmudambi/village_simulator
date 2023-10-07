from typing import Dict, List

import numpy as np
import pandas as pd
from vivarium import Component
from vivarium.framework.engine import Builder
from vivarium.framework.event import Event
from vivarium.framework.population import SimulantData
from vivarium.framework.time import Time

from village_simulator.simulation import sampling
from village_simulator.simulation.constants import ONE_YEAR
from village_simulator.simulation.utilities import get_annual_time_stamp


class Weather(Component):
    """Component that manages the weather"""

    CONFIGURATION_DEFAULTS = {
        "weather": {
            "temperature": {
                "seasonality": {
                    "min": 50,
                    "max": 80,
                    "min_date": {"month": 1, "day": 15},
                },
                "stochastic_variability": 5.0,
                "local_variability": 2.0,
            },
            "rainfall": {
                "distribution": sampling.STRETCHED_TRUNCNORM,
                "loc": {
                    "distribution": sampling.ZERO_INFLATED_GAMMA,
                    "zero_probability": 0.65,
                    "shape": 0.9902,
                    "scale": 10.0,
                },
                "scale": 0.1,
            },
        }
    }

    ##############
    # Properties #
    ##############

    @property
    def columns_created(self) -> List[str]:
        return ["temperature", "rainfall"]

    @property
    def initialization_requirements(self) -> Dict[str, List[str]]:
        return {"requires_streams": [self.name]}

    #####################
    # Lifecycle methods #
    #####################

    def setup(self, builder: Builder) -> None:
        self.configuration = builder.configuration.weather
        self.randomness = builder.randomness.get_stream(self.name)

        self.get_temperature = builder.value.register_value_producer(
            "temperature_distribution", self.temperature_source
        )

    def on_initialize_simulants(self, pop_data: SimulantData) -> None:
        self.population_view.update(
            pd.DataFrame(
                {column: np.nan for column in self.columns_created}, index=pop_data.index
            )
        )

    def on_time_step_prepare(self, event: Event) -> None:
        temperature = self.get_temperature(event)

        rainfall = sampling.from_configuration(
            self.configuration.rainfall, self.randomness, "rainfall", event.index
        )

        self.population_view.update(pd.concat([temperature, rainfall], axis=1))

    ####################
    # Pipeline sources #
    ####################

    def temperature_source(self, event: Event) -> pd.Series:
        config = self.configuration.temperature

        expected_temperature = self.get_expected_seasonal_temperature(event.time)
        regional_temperature_dist = sampling.FrozenDistribution(
            sampling.NORMAL,
            {"loc": expected_temperature, "scale": config.stochastic_variability},
            self.randomness,
            "regional_temperature",
        )

        composite_distribution = sampling.FrozenDistribution(
            sampling.NORMAL,
            {"loc": regional_temperature_dist, "scale": config.local_variability},
            self.randomness,
            "temperature",
            event.index,
        )
        return composite_distribution.sample()

    def get_expected_seasonal_temperature(self, time: Time) -> float:
        config = self.configuration.temperature.seasonality

        min_day = get_annual_time_stamp(time.year, config.min_date)
        distance_from_minimum = (time - min_day) / ONE_YEAR

        mean_temp = (config.max + config.min) * 0.5
        amplitude = 0.5 * (config.max - config.min)

        temperature = mean_temp - amplitude * np.cos(2 * np.pi * distance_from_minimum)
        return temperature
