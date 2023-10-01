from typing import List, Optional

import pandas as pd
from scipy import stats

from vivarium import Component, ConfigTree
from vivarium.framework.engine import Builder
from vivarium.framework.event import Event
from vivarium.framework.population import SimulantData
from vivarium.framework.randomness import RandomnessStream
from vivarium.framework.values import Pipeline

POPULATION_SIZE = "population_size"


class Demographics(Component):
    """
    Component that manages the population demographics of villages in the game.
    """

    CONFIGURATION_DEFAULTS = {
        "demographics": {
            "initial_village_size": 1_000_000,
            "mortality_rate": {
                "mean": 0.1,
                "standard_deviation": 0.01
            },
            "fertility_rate": {
                "mean": 0.15,
                "standard_deviation": 0.01
            },
        }
    }

    ##############
    # Properties #
    ##############

    @property
    def columns_created(self) -> List[str]:
        return [POPULATION_SIZE]

    #####################
    # Lifecycle methods #
    #####################

    def __init__(self):
        super().__init__()
        self.configuration: Optional[ConfigTree] = None
        self.randomness: Optional[RandomnessStream] = None
        self.mortality_rate: Optional[Pipeline] = None
        self.fertility_rate: Optional[Pipeline] = None

    def setup(self, builder: Builder) -> None:
        self.configuration = builder.configuration.demographics
        self.randomness = builder.randomness.get_stream(self.name)
        self.mortality_rate = builder.value.register_rate_producer(
            "mortality_rate",
            self.get_mortality_rate,
            requires_streams=[self.name]
        )
        self.fertility_rate = builder.value.register_rate_producer(
            "fertility_rate",
            self.get_fertility_rate,
            requires_streams=[self.name]
        )

    ########################
    # Event-driven methods #
    ########################

    def on_initialize_simulants(self, pop_data: SimulantData) -> None:
        self.population_view.update(
            pd.Series(self.configuration.initial_village_size, index=pop_data.index)
        )

    def on_time_step(self, event: Event) -> None:
        villages = self.population_view.get(event.index)
        population_size = villages[POPULATION_SIZE]

        fertility_rate = self.fertility_rate(villages.index)
        mortality_rate = self.mortality_rate(villages.index)
        population_size *= 1.0 + fertility_rate - mortality_rate

        population_size = population_size.round().astype(int)
        self.population_view.update(population_size)

    ####################
    # Pipeline sources #
    ####################

    def get_mortality_rate(self, index: pd.Index) -> pd.Series:
        mortality_rate = stats.norm.ppf(
            self.randomness.get_draw(index, "mortality"),
            loc=self.configuration.mortality_rate.mean,
            scale=self.configuration.mortality_rate.standard_deviation,
        )
        return pd.Series(mortality_rate)

    def get_fertility_rate(self, index: pd.Index) -> pd.Series:
        fertility_rate = stats.norm.ppf(
            self.randomness.get_draw(index, "fertility"),
            loc=self.configuration.fertility_rate.mean,
            scale=self.configuration.fertility_rate.standard_deviation,
        )
        return pd.Series(fertility_rate)
