from typing import Any, Dict, List, Optional

import pandas as pd
from vivarium import Component
from vivarium.framework.engine import Builder
from vivarium.framework.event import Event
from vivarium.framework.population import SimulantData

from village_simulator.simulation.map import FEATURE
from village_simulator.simulation.sampling import sample_from_distribution
from village_simulator.simulation.utilities import get_next_annual_event_date


class Resource(Component):
    """
    Component that manages a resource in the game.
    """

    CONFIGURATION_DEFAULTS = {
        "resource": {
            "initial_per_capita_stores": {"distribution": "normal", "loc": 1.0, "scale": 0.1},
            "annual_per_capita_consumption": {
                "distribution": "normal",
                "loc": 1.0,
                "scale": 0.1,
            },
            "annual_per_capita_accumulation": {
                "distribution": "normal",
                "loc": 1.0,
                "scale": 0.1,
            },
        }
    }

    ##############
    # Properties #
    ##############

    @property
    def configuration_defaults(self) -> Dict[str, Any]:
        return {self.resource: self.CONFIGURATION_DEFAULTS["resource"]}

    @property
    def columns_created(self) -> List[str]:
        return [self.resource_stores]

    @property
    def columns_required(self) -> List[str]:
        return [FEATURE]

    @property
    def initialization_requirements(self) -> Dict[str, List[str]]:
        return {"requires_values": ["total_population"], "requires_streams": [self.name]}

    @property
    def population_view_query(self) -> Optional[str]:
        return f"{FEATURE} == 'village'"

    #####################
    # Lifecycle methods #
    #####################

    def __init__(self, resource: str):
        super().__init__()
        self.resource = resource
        self.resource_stores = f"{self.resource}_stores"

    def setup(self, builder: Builder) -> None:
        self.configuration = builder.configuration[self.resource]
        self.initial_village_size = (
            builder.configuration.demographics.initial_village_size.loc
        )
        self.randomness = builder.randomness.get_stream(self.name)
        self.total_population = builder.value.get_value("total_population")

        self.consumption = self.register_consumption(builder)
        self.accumulation = self.register_accumulation(builder)

    #################
    # Setup methods #
    #################

    def register_accumulation(self, builder):
        return builder.value.register_rate_producer(
            f"{self.resource}.accumulation",
            self.get_accumulation_rate,
            requires_values=["total_population"],
            requires_streams=[self.name],
        )

    def register_consumption(self, builder):
        return builder.value.register_rate_producer(
            f"{self.resource}.consumption",
            self.get_consumption_rate,
            requires_values=["total_population"],
            requires_streams=[self.name],
        )

    ########################
    # Event-driven methods #
    ########################

    def on_initialize_simulants(self, pop_data: SimulantData) -> None:
        """
        Initialize resource stores.

        :param pop_data:
        :return:
        """
        feature = self.population_view.subview([FEATURE]).get(pop_data.index)
        village_index = feature[feature == "village"].index
        stores = pd.Series(0.0, index=pop_data.index, name=self.resource_stores)

        stores[village_index] = self.initial_village_size * sample_from_distribution(
            self.configuration.initial_per_capita_stores,
            self.randomness,
            self.resource,
            village_index,
        )
        self.population_view.update(stores)

    def on_time_step(self, event: Event) -> None:
        """
        Consume and accumulate resource.

        :param event:
        :return:
        """
        resource = self.population_view.get(event.index)[self.resource_stores]
        resource -= self.consumption(event.index)
        resource += self.accumulation(event.index)
        self.population_view.update(resource)

    ####################
    # Pipeline sources #
    ####################

    def get_consumption_rate(self, index: pd.Index) -> pd.Series:
        """
        Gets the rate at which the resource is consumed by each village.

        This is an annual rate, which will be rescaled to the time-step by the
        pipeline's post-processor.

        :param index:
        :return:
        """
        consumption_per_capita = sample_from_distribution(
            self.configuration.annual_per_capita_consumption,
            self.randomness,
            f"{self.resource}.consumption",
            index,
        )
        return self.get_total_from_per_capita(consumption_per_capita)

    def get_accumulation_rate(self, index: pd.Index) -> pd.Series:
        """
        Gets the rate at which the resource is accumulated by each village.

        This is an annual rate, which will be rescaled to the time-step by the
        pipeline's post-processor.

        :param index:
        :return:
        """
        accumulation_per_capita = sample_from_distribution(
            self.configuration.annual_per_capita_accumulation,
            self.randomness,
            f"{self.resource}.accumulation",
            index,
        )
        return self.get_total_from_per_capita(accumulation_per_capita)

    ##################
    # Helper methods #
    ##################

    def get_total_from_per_capita(self, per_capita_value: pd.Series) -> pd.Series:
        """Scales the per capita value to a raw value."""
        return self.total_population(per_capita_value.index) * per_capita_value


class Food(Resource):
    """
    Component that manages the food resources of villages in the game.
    """

    CONFIGURATION_DEFAULTS = {
        "food": {
            "initial_per_capita_stores": {
                "distribution": "normal",
                "loc": 10.0,
                "scale": 0.5,
            },
            "annual_per_capita_consumption": {
                "distribution": "normal",
                "loc": 10.0,
                "scale": 0.5,
            },
            "annual_per_capita_accumulation": {
                "distribution": "normal",
                "loc": 10.0,
                "scale": 0.5,
            },
            "harvest_date": {"month": 9, "day": 15},
        }
    }

    ##############
    # Properties #
    ##############

    @property
    def configuration_defaults(self) -> Dict[str, Any]:
        return self.CONFIGURATION_DEFAULTS

    #####################
    # Lifecycle methods #
    #####################

    def __init__(self):
        super().__init__("food")

    def setup(self, builder: Builder) -> None:
        super().setup(builder)
        self.clock = builder.time.clock()
        self.step_size = builder.time.step_size()

    #################
    # Setup methods #
    #################

    def register_accumulation(self, builder):
        return builder.value.register_value_producer(
            f"{self.resource}.accumulation",
            self.get_harvest_quantity,
            requires_values=["total_population"],
            requires_streams=[self.name],
        )

    ####################
    # Pipeline sources #
    ####################

    def get_harvest_quantity(self, index: pd.Index) -> pd.Series:
        """
        Gets the total amount of food accumulated during this time-step.

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
            harvest_per_capita = sample_from_distribution(
                self.configuration.base_harvest_per_capita,
                self.randomness,
                f"{self.resource}.accumulation",
                index,
            )
            harvest_quantity = self.get_total_from_per_capita(harvest_per_capita)
        else:
            harvest_quantity = pd.Series(
                0.0, index=index, name=f"{self.resource}.accumulation"
            )

        return harvest_quantity
