from typing import Dict, List

import numpy as np
import pandas as pd
from vivarium import Component
from vivarium.framework.engine import Builder
from vivarium.framework.event import Event
from vivarium.framework.population import SimulantData

from village_simulator.simulation import sampling


class Weather(Component):
    """Component that manages the weather"""

    CONFIGURATION_DEFAULTS = {
        "weather": {
            "temperature_fahrenheit": {
                "distribution": "normal",
                "loc": {
                    "distribution": "normal",
                    "loc": 65.5,
                    "scale": 15.0,
                },
                "scale": 3.0,
            },
            "rainfall": {
                "distribution": "stretched_truncnorm",
                "loc": {
                    "distribution": "zero_inflated_gamma",
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
        return ["temperature", "daily_rainfall"]

    @property
    def initialization_requirements(self) -> Dict[str, List[str]]:
        return {"requires_streams": [self.name]}

    #####################
    # Lifecycle methods #
    #####################

    def setup(self, builder: Builder) -> None:
        self.configuration = builder.configuration.weather
        self.randomness = builder.randomness.get_stream(self.name)

    def on_initialize_simulants(self, pop_data: SimulantData) -> None:
        self.population_view.update(
            pd.DataFrame(
                {column: np.nan for column in self.columns_created}, index=pop_data.index
            )
        )

    def on_time_step_prepare(self, event: Event) -> None:
        temperature = sampling.from_configuration(
            self.configuration.temperature_fahrenheit,
            self.randomness,
            "temperature",
            event.index,
        )

        rainfall = sampling.from_configuration(
            self.configuration.rainfall, self.randomness, "daily_rainfall", event.index
        )

        self.population_view.update(pd.concat([temperature, rainfall], axis=1))
