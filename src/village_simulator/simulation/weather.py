from typing import Dict, List

import numpy as np
import pandas as pd
from vivarium import Component
from vivarium.framework.engine import Builder
from vivarium.framework.event import Event
from vivarium.framework.population import SimulantData

from village_simulator.simulation import sampling
from village_simulator.simulation.utilities import get_value_from_annual_cycle


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
                "seasonality": {
                    "min": 0.1,
                    "max": 1.0,
                    "min_date": {"month": 2, "day": 15},
                },
                "dry_probability": 0.55,
                "gamma_shape_parameter": 0.9902,
                "gamma_scale_parameter": 15.0,
                "local_variability": 0.05,
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
            "temperature", self.temperature_source
        )

        self.get_rainfall = builder.value.register_value_producer(
            "rainfall", self.rainfall_source
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

        expected_temperature = get_value_from_annual_cycle(config.seasonality, event.time)
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

    def rainfall_source(self, event: Event) -> pd.Series:
        config = self.configuration.rainfall

        aridity_factor = get_value_from_annual_cycle(config.seasonality, event.time)
        dry_probability = 1 - aridity_factor * (1 - config.dry_probability)
        scale = aridity_factor * config.gamma_scale_parameter

        regional_rainfall_dist = sampling.FrozenDistribution(
            sampling.ZERO_INFLATED_GAMMA,
            {
                "zero_probability": dry_probability,
                "shape": config.gamma_shape_parameter,
                "scale": scale,
            },
            self.randomness,
            "regional_rainfall",
        )

        composite_distribution = sampling.FrozenDistribution(
            sampling.STRETCHED_TRUNCNORM,
            {"loc": regional_rainfall_dist, "scale": config.local_variability},
            self.randomness,
            "rainfall",
            event.index,
        )
        return composite_distribution.sample()
