from typing import List

import pandas as pd

from vivarium import Component
from vivarium.framework.engine import Builder
from vivarium.framework.event import Event
from vivarium.framework.population import SimulantData


POPULATION_SIZE = "population_size"


class Demographics(Component):
    """
    Component that manages the population demographics of villages in the game.
    """

    CONFIGURATION_DEFAULTS = {
        "demographics": {
            "initial_village_size": 1000,
            "mortality_rate": 0.1,
            "fertility_rate": 0.15,
        }
    }

    @property
    def columns_created(self) -> List[str]:
        return [POPULATION_SIZE]

    def __init__(self):
        super().__init__()
        self.configuration = None
        self.mortality_rate = None
        self.fertility_rate = None

    def setup(self, builder: Builder) -> None:
        self.configuration = builder.configuration.demographics
        self.mortality_rate = builder.value.register_rate_producer(
            "mortality_rate",
            lambda index: pd.Series(self.configuration.mortality_rate, index=index)
        )
        self.fertility_rate = builder.value.register_rate_producer(
            "fertility_rate",
            lambda index: pd.Series(self.configuration.fertility_rate, index=index)
        )

    def on_initialize_simulants(self, pop_data: SimulantData) -> None:
        self.population_view.update(
            pd.Series(self.configuration.initial_village_size, index=pop_data.index)
        )

    def on_time_step(self, event: Event) -> None:
        villages = self.population_view.get(event.index)
        population_size = villages[POPULATION_SIZE]
        population_size *= (
            1 + self.fertility_rate(villages.index) - self.mortality_rate(villages.index)
        )
        population_size = population_size.round().astype(int)
        self.population_view.update(population_size)
