from typing import Any, Dict, List, Optional

import pandas as pd
from scipy import stats
from vivarium import Component
from vivarium.framework.engine import Builder
from vivarium.framework.event import Event
from vivarium.framework.population import SimulantData

from village_simulator.constants import Columns, Pipelines


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
        return [self._stores_column]

    @property
    def columns_required(self) -> Optional[List[str]]:
        return [Columns.IS_VILLAGE]

    @property
    def initialization_requirements(self) -> Dict[str, List[str]]:
        return {
            "requires_columns": [Columns.IS_VILLAGE],
            "requires_values": [Pipelines.TOTAL_POPULATION],
            "requires_streams": [self.name],
        }

    @property
    def population_view_query(self) -> Optional[str]:
        return f"{Columns.IS_VILLAGE} == True"

    #####################
    # Lifecycle methods #
    #####################

    def __init__(self, resource: str):
        super().__init__()
        self.resource = resource
        self._stores_column = Columns.get_resource_stores(self.resource)
        self._consumption_pipeline = Pipelines.get_resource_consumption(self.resource)
        self._accumulation_pipeline = Pipelines.get_resource_accumulation(self.resource)

    def setup(self, builder: Builder) -> None:
        self.configuration = builder.configuration[self.resource]
        self.randomness = builder.randomness.get_stream(self.name)
        self.total_population = builder.value.get_value(Pipelines.TOTAL_POPULATION)

        self.get_consumption = self.register_consumption(builder)
        self.get_accumulation = self.register_accumulation(builder)

    #################
    # Setup methods #
    #################

    def register_accumulation(self, builder):
        return builder.value.register_rate_producer(
            self._accumulation_pipeline,
            self.accumulation_rate_source,
            requires_values=[Pipelines.TOTAL_POPULATION],
            requires_streams=[self.name],
        )

    def register_consumption(self, builder):
        return builder.value.register_rate_producer(
            self._consumption_pipeline,
            self.consumption_rate_source,
            requires_values=[Pipelines.TOTAL_POPULATION],
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
        is_village = self.population_view.subview([Columns.IS_VILLAGE]).get(pop_data.index)
        village_index = is_village[is_village].index
        stores = pd.Series(0.0, index=pop_data.index, name=self._stores_column)

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
        resource = self.population_view.get(event.index)[self._stores_column]
        resource -= self.get_consumption(resource.index)
        resource += self.get_accumulation(resource.index)
        self.population_view.update(resource)

    ####################
    # Pipeline sources #
    ####################

    def accumulation_rate_source(self, index: pd.Index) -> pd.Series:
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

    def consumption_rate_source(self, index: pd.Index) -> pd.Series:
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
