from typing import Dict, List, Optional

import pandas as pd
from scipy import stats
from vivarium import Component
from vivarium.framework.engine import Builder
from vivarium.framework.event import Event
from vivarium.framework.population import SimulantData

from village_simulator.constants import Columns, Pipelines
from village_simulator.simulation.utilities import round_stochastic


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
        return [Columns.FEMALE_POPULATION_SIZE, Columns.MALE_POPULATION_SIZE]

    @property
    def columns_required(self) -> List[str]:
        return [Columns.IS_VILLAGE, Columns.ARABLE_LAND]

    @property
    def initialization_requirements(self) -> Dict[str, List[str]]:
        return {
            "requires_columns": [Columns.IS_VILLAGE, Columns.ARABLE_LAND],
            "requires_streams": [self.name],
        }

    @property
    def population_view_query(self) -> Optional[str]:
        return f"{Columns.IS_VILLAGE} == True"

    #####################
    # Lifecycle methods #
    #####################

    def setup(self, builder: Builder) -> None:
        self.configuration = builder.configuration.demographics
        self.randomness = builder.randomness.get_stream(self.name)
        self.get_fertility_rate = builder.value.register_rate_producer(
            Pipelines.FERTILITY_RATE, self.fertility_rate_source, requires_streams=[self.name]
        )
        self.get_mortality_rate = builder.value.register_rate_producer(
            Pipelines.MORTALITY_RATE, self.mortality_rate_source, requires_streams=[self.name]
        )
        self.get_total_population = builder.value.register_value_producer(
            Pipelines.TOTAL_POPULATION,
            self.total_population_source,
            requires_columns=[Columns.FEMALE_POPULATION_SIZE, Columns.MALE_POPULATION_SIZE],
        )

    ########################
    # Event-driven methods #
    ########################

    def on_initialize_simulants(self, pop_data: SimulantData) -> None:
        state = self.population_view.subview([Columns.ARABLE_LAND]).get(pop_data.index)
        village_size = pd.DataFrame(
            0,
            index=pop_data.index,
            columns=[Columns.FEMALE_POPULATION_SIZE, Columns.MALE_POPULATION_SIZE],
        )

        total_village_size = (
            self.randomness.sample_from_distribution(
                state.index,
                distribution=stats.norm,
                additional_key="initial_village_size",
                **self.configuration.initial_village_size.to_dict(),
            )
            * state[Columns.ARABLE_LAND]
        )

        sex_ratio = self.randomness.sample_from_distribution(
            state.index,
            distribution=stats.norm,
            additional_key="initial_sex_ratio",
            **self.configuration.initial_sex_ratio.to_dict(),
        )

        village_size.loc[state.index, Columns.FEMALE_POPULATION_SIZE] = round_stochastic(
            total_village_size * sex_ratio / 2.0,
            self.randomness,
            "initial_female_village_size",
        )
        village_size.loc[state.index, Columns.MALE_POPULATION_SIZE] = round_stochastic(
            total_village_size * (1.0 - sex_ratio / 2.0),
            self.randomness,
            "initial_male_village_size",
        )

        self.population_view.update(village_size)

    def on_time_step(self, event: Event) -> None:
        villages = self.population_view.get(event.index)[
            [Columns.FEMALE_POPULATION_SIZE, Columns.MALE_POPULATION_SIZE]
        ]

        fertility_rate = self.get_fertility_rate(villages.index)
        births = fertility_rate.mul(villages[Columns.FEMALE_POPULATION_SIZE], axis=0)

        mortality_rate = self.get_mortality_rate(villages.index)
        deaths = mortality_rate * villages
        villages += births - deaths

        villages = round_stochastic(villages, self.randomness, "population_size")
        self.population_view.update(villages)

    ####################
    # Pipeline sources #
    ####################

    def fertility_rate_source(self, index: pd.Index) -> pd.DataFrame:
        """
        Get the sex-specific fertility rate for the population in the given
        index.

        :param index:
        :return:
        """
        # FIXME: The names of the columns of the returned DataFrame are just the
        #  names of the respective sex-specific population size columns. This is
        #  confusing and will likely lead to bugs in the future if not fixed.
        female_fertility_rate = self.randomness.sample_from_distribution(
            index,
            distribution=stats.norm,
            additional_key="female_fertility",
            **self.configuration.fertility_rate.to_dict(),
        ).rename(Columns.FEMALE_POPULATION_SIZE)

        male_fertility_rate = self.randomness.sample_from_distribution(
            index,
            distribution=stats.norm,
            additional_key="male_fertility",
            **self.configuration.fertility_rate.to_dict(),
        ).rename(Columns.MALE_POPULATION_SIZE)
        return pd.concat([female_fertility_rate, male_fertility_rate], axis=1)

    def mortality_rate_source(self, index: pd.Index) -> pd.DataFrame:
        female_mortality_rate = self.randomness.sample_from_distribution(
            index,
            distribution=stats.norm,
            additional_key="female_mortality",
            **self.configuration.mortality_rate.to_dict(),
        ).rename(Columns.FEMALE_POPULATION_SIZE)
        male_mortality_rate = self.randomness.sample_from_distribution(
            index,
            distribution=stats.norm,
            additional_key="male_mortality",
            **self.configuration.mortality_rate.to_dict(),
        ).rename(Columns.MALE_POPULATION_SIZE)
        return pd.concat([female_mortality_rate, male_mortality_rate], axis=1)

    def total_population_source(self, index: pd.Index) -> pd.Series:
        population = self.population_view.get(index)[
            [Columns.FEMALE_POPULATION_SIZE, Columns.MALE_POPULATION_SIZE]
        ]
        return population.sum(axis=1)
