from typing import Dict, List, Optional

import pandas as pd
from vivarium import Component, ConfigTree
from vivarium.framework.engine import Builder
from vivarium.framework.event import Event
from vivarium.framework.population import SimulantData
from vivarium.framework.randomness import RandomnessStream
from vivarium.framework.values import Pipeline

from village_simulator.simulation.utilities import (
    round_stochastic,
    sample_from_normal_distribution,
)

FEMALE_POPULATION_SIZE = "female_population_size"
MALE_POPULATION_SIZE = "male_population_size"


class Demographics(Component):
    """
    Component that manages the population demographics of villages in the game.
    """

    CONFIGURATION_DEFAULTS = {
        "demographics": {
            "initial_village_size": {"mean": 1_000_000, "standard_deviation": 100},
            "initial_sex_ratio": {"mean": 1.0, "standard_deviation": 0.01},
            "fertility_rate": {"mean": 0.15, "standard_deviation": 0.01},
            "mortality_rate": {"mean": 0.1, "standard_deviation": 0.01},
        }
    }

    ##############
    # Properties #
    ##############

    @property
    def columns_created(self) -> List[str]:
        return [FEMALE_POPULATION_SIZE, MALE_POPULATION_SIZE]

    @property
    def initialization_requirements(self) -> Dict[str, List[str]]:
        return {"requires_streams": [self.name]}

    #####################
    # Lifecycle methods #
    #####################

    def __init__(self):
        super().__init__()
        self.configuration: Optional[ConfigTree] = None
        self.randomness: Optional[RandomnessStream] = None
        self.fertility_rate: Optional[Pipeline] = None
        self.mortality_rate: Optional[Pipeline] = None

    def setup(self, builder: Builder) -> None:
        self.configuration = builder.configuration.demographics
        self.randomness = builder.randomness.get_stream(self.name)
        self.fertility_rate = builder.value.register_rate_producer(
            "fertility_rate", self.get_fertility_rate, requires_streams=[self.name]
        )
        self.mortality_rate = builder.value.register_rate_producer(
            "mortality_rate", self.get_mortality_rate, requires_streams=[self.name]
        )

    ########################
    # Event-driven methods #
    ########################

    def on_initialize_simulants(self, pop_data: SimulantData) -> None:
        village_size = sample_from_normal_distribution(
            pop_data.index,
            self.configuration.initial_village_size,
            self.randomness,
            "initial_village_size",
        )

        sex_ratio = sample_from_normal_distribution(
            pop_data.index,
            self.configuration.initial_sex_ratio,
            self.randomness,
            "initial_sex_ratio",
        )

        female_village_size = round_stochastic(
            village_size * sex_ratio / 2.0, self.randomness, "initial_female_village_size"
        ).rename(FEMALE_POPULATION_SIZE)
        male_village_size = round_stochastic(
            village_size * (1.0 - sex_ratio / 2.0),
            self.randomness,
            "initial_male_village_size",
        ).rename(MALE_POPULATION_SIZE)

        self.population_view.update(
            pd.concat([female_village_size, male_village_size], axis=1)
        )

    def on_time_step(self, event: Event) -> None:
        villages = self.population_view.get(event.index)[
            [FEMALE_POPULATION_SIZE, MALE_POPULATION_SIZE]
        ]

        fertility_rate = self.fertility_rate(villages.index)
        mortality_rate = self.mortality_rate(villages.index)
        villages *= 1.0 + fertility_rate - mortality_rate

        villages = round_stochastic(villages, self.randomness, "population_size")
        self.population_view.update(villages)

    ####################
    # Pipeline sources #
    ####################

    def get_fertility_rate(self, index: pd.Index) -> pd.DataFrame:
        female_fertility_rate = sample_from_normal_distribution(
            index, self.configuration.fertility_rate, self.randomness, "female_fertility"
        ).rename(FEMALE_POPULATION_SIZE)
        male_fertility_rate = sample_from_normal_distribution(
            index, self.configuration.fertility_rate, self.randomness, "male_fertility"
        ).rename(MALE_POPULATION_SIZE)
        return pd.concat([female_fertility_rate, male_fertility_rate], axis=1)

    def get_mortality_rate(self, index: pd.Index) -> pd.DataFrame:
        female_mortality_rate = sample_from_normal_distribution(
            index, self.configuration.mortality_rate, self.randomness, "female_mortality"
        ).rename(FEMALE_POPULATION_SIZE)
        male_mortality_rate = sample_from_normal_distribution(
            index, self.configuration.mortality_rate, self.randomness, "male_mortality"
        ).rename(MALE_POPULATION_SIZE)
        return pd.concat([female_mortality_rate, male_mortality_rate], axis=1)
