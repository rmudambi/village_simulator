from typing import Dict, List, Optional

import pandas as pd
from scipy import stats
from vivarium import Component
from vivarium.framework.engine import Builder
from vivarium.framework.event import Event
from vivarium.framework.population import SimulantData

from village_simulator.simulation.components.village import IS_VILLAGE, ARABLE_LAND
from village_simulator.simulation.utilities import round_stochastic

FEMALE_POPULATION_SIZE = "female_population_size"
MALE_POPULATION_SIZE = "male_population_size"


class Demographics(Component):
    """
    Component that manages the population demographics of villages in the game.
    """

    CONFIGURATION_DEFAULTS = {
        "demographics": {
            "initial_village_size": {"loc": 1_000, "scale": 1.0},
            "initial_sex_ratio": {"loc": 1.0, "scale": 0.01},
            "fertility_rate": {"loc": 0.05, "scale": 0.01},
            "mortality_rate": {"loc": 0.04, "scale": 0.01},
        }
    }

    ##############
    # Properties #
    ##############

    @property
    def columns_created(self) -> List[str]:
        return [FEMALE_POPULATION_SIZE, MALE_POPULATION_SIZE]

    @property
    def columns_required(self) -> List[str]:
        return [IS_VILLAGE, ARABLE_LAND]

    @property
    def initialization_requirements(self) -> Dict[str, List[str]]:
        return {"requires_columns": [IS_VILLAGE, ARABLE_LAND], "requires_streams": [self.name]}

    @property
    def population_view_query(self) -> Optional[str]:
        return f"{IS_VILLAGE} == True"

    #####################
    # Lifecycle methods #
    #####################

    def setup(self, builder: Builder) -> None:
        self.configuration = builder.configuration.demographics
        self.randomness = builder.randomness.get_stream(self.name)
        self.fertility_rate = builder.value.register_rate_producer(
            "fertility_rate", self.get_fertility_rate, requires_streams=[self.name]
        )
        self.mortality_rate = builder.value.register_rate_producer(
            "mortality_rate", self.get_mortality_rate, requires_streams=[self.name]
        )
        self.total_population = builder.value.register_value_producer(
            "total_population",
            self.get_total_population,
            requires_columns=[FEMALE_POPULATION_SIZE, MALE_POPULATION_SIZE],
        )

    ########################
    # Event-driven methods #
    ########################

    def on_initialize_simulants(self, pop_data: SimulantData) -> None:
        state = self.population_view.subview([ARABLE_LAND]).get(pop_data.index)
        female_village_size = pd.Series(0, index=pop_data.index, name=FEMALE_POPULATION_SIZE)
        male_village_size = pd.Series(0, index=pop_data.index, name=MALE_POPULATION_SIZE)

        village_size = state[ARABLE_LAND] * self.randomness.sample_from_distribution(
            state.index,
            distribution=stats.norm,
            additional_key="initial_village_size",
            **self.configuration.initial_village_size.to_dict(),
        )

        sex_ratio = self.randomness.sample_from_distribution(
            state.index,
            distribution=stats.norm,
            additional_key="initial_sex_ratio",
            **self.configuration.initial_sex_ratio.to_dict(),
        )

        female_village_size[state.index] = round_stochastic(
            village_size * sex_ratio / 2.0, self.randomness, "initial_female_village_size"
        )
        male_village_size[state.index] = round_stochastic(
            village_size * (1.0 - sex_ratio / 2.0),
            self.randomness,
            "initial_male_village_size",
        )

        self.population_view.update(
            pd.concat([female_village_size, male_village_size], axis=1)
        )

    def on_time_step(self, event: Event) -> None:
        villages = self.population_view.get(event.index)[
            [FEMALE_POPULATION_SIZE, MALE_POPULATION_SIZE]
        ]

        fertility_rate = self.fertility_rate(villages.index)
        births = fertility_rate.mul(villages[FEMALE_POPULATION_SIZE], axis=0)

        mortality_rate = self.mortality_rate(villages.index)
        deaths = mortality_rate * villages
        villages += births - deaths

        villages = round_stochastic(villages, self.randomness, "population_size")
        self.population_view.update(villages)

    ####################
    # Pipeline sources #
    ####################

    def get_fertility_rate(self, index: pd.Index) -> pd.DataFrame:
        female_fertility_rate = self.randomness.sample_from_distribution(
            index,
            distribution=stats.norm,
            additional_key="female_fertility",
            **self.configuration.fertility_rate.to_dict(),
        ).rename(FEMALE_POPULATION_SIZE)

        male_fertility_rate = self.randomness.sample_from_distribution(
            index,
            distribution=stats.norm,
            additional_key="male_fertility",
            **self.configuration.fertility_rate.to_dict(),
        ).rename(MALE_POPULATION_SIZE)
        return pd.concat([female_fertility_rate, male_fertility_rate], axis=1)

    def get_mortality_rate(self, index: pd.Index) -> pd.DataFrame:
        female_mortality_rate = self.randomness.sample_from_distribution(
            index,
            distribution=stats.norm,
            additional_key="female_mortality",
            **self.configuration.mortality_rate.to_dict(),
        ).rename(FEMALE_POPULATION_SIZE)
        male_mortality_rate = self.randomness.sample_from_distribution(
            index,
            distribution=stats.norm,
            additional_key="male_mortality",
            **self.configuration.mortality_rate.to_dict(),
        ).rename(MALE_POPULATION_SIZE)
        return pd.concat([female_mortality_rate, male_mortality_rate], axis=1)

    def get_total_population(self, index: pd.Index) -> pd.Series:
        population = self.population_view.get(index)[
            [FEMALE_POPULATION_SIZE, MALE_POPULATION_SIZE]
        ]
        return population.sum(axis=1).rename("total_population")
