from typing import Dict, List

import pandas as pd
from vivarium import Component
from vivarium.framework.engine import Builder
from vivarium.framework.event import Event
from vivarium.framework.population import SimulantData

from village_simulator.simulation.utilities import (
    get_next_annual_event_date,
    sample_from_normal_distribution,
)

FOOD = "food"
WOOD = "wood"


class Food(Component):
    """
    Component that manages the food resources of villages in the game.
    """

    CONFIGURATION_DEFAULTS = {
        "food": {
            "initial_per_capita_food_stores": {"mean": 1_000.0, "standard_deviation": 5.0},
            "annual_per_capita_food_consumption": {
                "mean": 1_000.0,
                "standard_deviation": 5.0,
            },
            "harvest_date": {"month": 9, "day": 15},
            "base_harvest_per_capita": {"mean": 1_000.0, "standard_deviation": 5.0},
        }
    }

    ##############
    # Properties #
    ##############

    @property
    def columns_created(self) -> List[str]:
        return [FOOD]

    @property
    def initialization_requirements(self) -> Dict[str, List[str]]:
        return {"requires_values": ["total_population"], "requires_streams": [self.name]}

    #####################
    # Lifecycle methods #
    #####################

    def setup(self, builder: Builder) -> None:
        self.clock = builder.time.clock()
        self.step_size = builder.time.step_size()
        self.configuration = builder.configuration.food
        self.initial_village_size = (
            builder.configuration.demographics.initial_village_size.mean
        )
        self.randomness = builder.randomness.get_stream(self.name)
        self.total_population = builder.value.get_value("total_population")
        self.food_consumption_rate = builder.value.register_rate_producer(
            "food_consumption_rate",
            self.get_food_consumption_rate,
            requires_values=["total_population"],
            requires_streams=[self.name],
        )
        self.harvest_quantity = builder.value.register_value_producer(
            "harvest_quantity",
            self.get_harvest_quantity,
            requires_values=["total_population"],
            requires_streams=[self.name],
        )

    ########################
    # Event-driven methods #
    ########################

    def on_initialize_simulants(self, pop_data: SimulantData) -> None:
        """
        Initialize food stores. This value scales linearly with the expected
        initial village size.

        :param pop_data:
        :return:
        """
        food = self.initial_village_size * sample_from_normal_distribution(
            pop_data.index,
            self.configuration.initial_per_capita_food_stores,
            self.randomness,
            "initial_food_stores",
        ).rename(FOOD)
        self.population_view.update(food)

    def on_time_step(self, event: Event) -> None:
        """
        Consume food and harvest crops.

        :param event:
        :return:
        """
        food = self.population_view.get(event.index)[FOOD]
        food -= self.food_consumption_rate(event.index)
        food += self.harvest_quantity(event.index)
        self.population_view.update(food)

    ####################
    # Pipeline sources #
    ####################

    def get_food_consumption_rate(self, index: pd.Index) -> pd.Series:
        """
        Gets the rate at which food is consumed by each village.

        This is an annual rate, which will be rescaled to the time-step by the
        pipeline's post-processor

        :param index:
        :return:
        """
        food_consumption_per_capita = sample_from_normal_distribution(
            index,
            self.configuration.annual_per_capita_food_consumption,
            self.randomness,
            "food_consumption_rate",
        ).rename("food_consumption_rate")
        return self._get_total_from_per_capita(food_consumption_per_capita)

    def get_harvest_quantity(self, index: pd.Index) -> pd.Series:
        """
        Gets the total amount of food harvested during this time-step.

        Right now all food is harvested at a single time in the year

        :param index:
        :return:
        """
        clock_time = self.clock()
        next_harvest_date = get_next_annual_event_date(
            clock_time,
            self.configuration.harvest_date.month,
            self.configuration.harvest_date.day,
        )
        if clock_time < next_harvest_date <= clock_time + self.step_size():
            harvest_per_capita = sample_from_normal_distribution(
                index,
                self.configuration.base_harvest_per_capita,
                self.randomness,
                "base_harvest",
            ).rename("harvest_quantity")
            harvest_quantity = self._get_total_from_per_capita(harvest_per_capita)
        else:
            harvest_quantity = pd.Series(0.0, index=index, name="harvest_quantity")

        return harvest_quantity

    ##################
    # Helper methods #
    ##################

    def _get_total_from_per_capita(self, per_capita_value: pd.Series) -> pd.Series:
        """Scales the per capita value to a raw value."""
        return pd.Series(
            self.total_population(per_capita_value.index) * per_capita_value,
            name=per_capita_value.name
        )


class Wood(Component):
    """
    Component that manages the wood resources of villages in the game.
    """

    CONFIGURATION_DEFAULTS = {
        "wood": {
            "initial_per_capita_wood_stores": {"mean": 50.0, "standard_deviation": 10.0},
            "annual_per_capita_wood_consumption": {
                "mean": 15.0,
                "standard_deviation": 3.0,
            },
            "annual_per_capita_wood_accumulation": {"mean": 16.0, "standard_deviation": 5.0},
        }
    }

    ##############
    # Properties #
    ##############

    @property
    def columns_created(self) -> List[str]:
        return [WOOD]

    @property
    def initialization_requirements(self) -> Dict[str, List[str]]:
        return {"requires_values": ["total_population"], "requires_streams": [self.name]}

    #####################
    # Lifecycle methods #
    #####################

    def setup(self, builder: Builder) -> None:
        self.configuration = builder.configuration.wood
        self.initial_village_size = (
            builder.configuration.demographics.initial_village_size.mean
        )
        self.randomness = builder.randomness.get_stream(self.name)
        self.total_population = builder.value.get_value("total_population")

        self.wood_consumption_rate = builder.value.register_rate_producer(
            "wood_consumption_rate",
            self.get_wood_consumption_rate,
            requires_streams=[self.name],
        )
        self.wood_accumulation_rate = builder.value.register_value_producer(
            "wood_accumulation_rate",
            self.get_wood_accumulation_rate,
            requires_streams=[self.name],
        )

    ########################
    # Event-driven methods #
    ########################

    def on_initialize_simulants(self, pop_data: SimulantData) -> None:
        """
        Initialize wood stores.

        :param pop_data:
        :return:
        """
        wood = self.initial_village_size * sample_from_normal_distribution(
            pop_data.index,
            self.configuration.initial_per_capita_wood_stores,
            self.randomness,
            "initial_wood_stores",
        ).rename(WOOD)
        self.population_view.update(wood)

    def on_time_step(self, event: Event) -> None:
        """
        Consume and accumulate wood.

        :param event:
        :return:
        """
        wood = self.population_view.get(event.index)[WOOD]
        wood -= self.wood_consumption_rate(event.index)
        wood += self.wood_accumulation_rate(event.index)
        self.population_view.update(wood)

    ####################
    # Pipeline sources #
    ####################

    def get_wood_consumption_rate(self, index: pd.Index) -> pd.Series:
        """
        Gets the rate at which wood is consumed by each village.

        This is an annual rate, which will be rescaled to the time-step by the
        pipeline's post-processor

        :param index:
        :return:
        """
        wood_consumption_per_capita = sample_from_normal_distribution(
            index,
            self.configuration.annual_per_capita_wood_consumption,
            self.randomness,
            "wood_consumption_rate",
        ).rename("wood_consumption_rate")
        return self._get_total_from_per_capita(wood_consumption_per_capita)

    def get_wood_accumulation_rate(self, index: pd.Index) -> pd.Series:
        """
        Gets the total amount of wood accumulated during this time-step.

        :param index:
        :return:
        """
        wood_accumulation_per_capita = sample_from_normal_distribution(
            index,
            self.configuration.annual_per_capita_wood_accumulation,
            self.randomness,
            "wood_accumulation_rate",
        ).rename("wood_accumulation_rate")
        return self._get_total_from_per_capita(wood_accumulation_per_capita)

    ##################
    # Helper methods #
    ##################

    def _get_total_from_per_capita(self, per_capita_value: pd.Series) -> pd.Series:
        """Scales the per capita value to a raw value."""
        return pd.Series(
            self.total_population(per_capita_value.index) * per_capita_value,
            name=per_capita_value.name
        )
