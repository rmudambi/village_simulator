from typing import Any, Dict, List, Optional

import pandas as pd
from scipy import stats
from vivarium import Component
from vivarium.framework.engine import Builder
from vivarium.framework.event import Event
from vivarium.framework.population import SimulantData

from village_simulator.simulation.components.village import IS_VILLAGE


class Resource(Component):
    """
    Component that manages a resource in the game.
    """

    CONFIGURATION_DEFAULTS = {
        "resource": {
            "initial_per_capita_stores": {"loc": 1.0, "scale": 0.1},
            "annual_per_capita_consumption": {"loc": 1.0, "scale": 0.1},
            "annual_per_capita_accumulation": {"loc": 1.0, "scale": 0.1},
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
    def columns_required(self) -> Optional[List[str]]:
        return [IS_VILLAGE]

    @property
    def initialization_requirements(self) -> Dict[str, List[str]]:
        return {
            "requires_columns": [IS_VILLAGE],
            "requires_values": ["total_population"],
            "requires_streams": [self.name],
        }

    @property
    def population_view_query(self) -> Optional[str]:
        return f"{IS_VILLAGE} == True"

    #####################
    # Lifecycle methods #
    #####################

    def __init__(self, resource: str):
        super().__init__()
        self.resource = resource
        self.resource_stores = f"{self.resource}_stores"

    def setup(self, builder: Builder) -> None:
        self.configuration = builder.configuration[self.resource]
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
        is_village = self.population_view.subview([IS_VILLAGE]).get(pop_data.index)
        village_index = is_village[is_village].index
        stores = pd.Series(0.0, index=pop_data.index, name=self.resource_stores)

        stores[village_index] = self.total_population(
            village_index
        ) * self.randomness.sample_from_distribution(
            village_index,
            distribution=stats.norm,
            additional_key=self.resource,
            **self.configuration.initial_per_capita_stores.to_dict(),
        )

        self.population_view.update(stores)

    def on_time_step(self, event: Event) -> None:
        """
        Consume and accumulate resource.

        :param event:
        :return:
        """
        resource = self.population_view.get(event.index)[self.resource_stores]
        resource -= self.consumption(resource.index)
        resource += self.accumulation(resource.index)
        self.population_view.update(resource)

    ####################
    # Pipeline sources #
    ####################

    def get_accumulation_rate(self, index: pd.Index) -> pd.Series:
        """
        Gets the rate at which the resource is accumulated by each village.

        This is an annual rate, which will be rescaled to the time-step by the
        pipeline's post-processor.

        :param index:
        :return:
        """
        accumulation_per_capita = self.randomness.sample_from_distribution(
            index,
            distribution=stats.norm,
            additional_key=f"{self.resource}.accumulation",
            **self.configuration.annual_per_capita_accumulation.to_dict(),
        )
        return self.get_total_from_per_capita(accumulation_per_capita)

    def get_consumption_rate(self, index: pd.Index) -> pd.Series:
        """
        Gets the rate at which the resource is consumed by each village.

        This is an annual rate, which will be rescaled to the time-step by the
        pipeline's post-processor.

        :param index:
        :return:
        """
        consumption_per_capita = self.randomness.sample_from_distribution(
            index,
            distribution=stats.norm,
            additional_key=f"{self.resource}.consumption",
            **self.configuration.annual_per_capita_consumption.to_dict(),
        )
        return self.get_total_from_per_capita(consumption_per_capita)

    ##################
    # Helper methods #
    ##################

    def get_total_from_per_capita(self, per_capita_value: pd.Series) -> pd.Series:
        """Scales the per capita value to a raw value."""
        return self.total_population(per_capita_value.index) * per_capita_value
