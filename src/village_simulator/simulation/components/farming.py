from typing import List, Optional

import pandas as pd
from vivarium.framework.engine import Builder
from vivarium.framework.event import Event
from vivarium.framework.population import SimulantData
from vivarium.framework.time import get_time_stamp

from village_simulator.paths import EFFECT_OF_TEMPERATURE_ON_WHEAT_YIELD
from village_simulator.simulation import Resource
from village_simulator.simulation.components.map import FEATURE
from village_simulator.simulation.constants import ONE_YEAR
from village_simulator.simulation.utilities import get_next_annual_event_date


WHEAT_SOWING_DATE = {"month": 10, "day": 15}
WHEAT_HARVEST_DATE = {"month": 5, "day": 15}


class Wheat(Resource):
    """
    Component that manages the food resources of villages in the game.
    """

    CONFIGURATION_DEFAULTS = {
        "resource": {
            "initial_per_capita_stores": {"loc": 10.0, "scale": 0.5},
            "annual_per_capita_consumption": {"loc": 10.0, "scale": 0.5},
            "annual_per_capita_accumulation": {"loc": 10.0, "scale": 0.5},
            "sowing_date": WHEAT_SOWING_DATE,
            "harvest_date": WHEAT_HARVEST_DATE,
        }
    }

    ##############
    # Properties #
    ##############

    @property
    def columns_created(self) -> List[str]:
        return super().columns_created + [
            self.projected_yield_column,
            self.previously_dry_column,
            self.cumulative_dry_days_column,
            self.rainfall_mid_growth_column,
            self.rainfall_late_growth_column,
        ]

    @property
    def columns_required(self) -> Optional[List[str]]:
        return super().columns_required + ["temperature", "rainfall"]

    #####################
    # Lifecycle methods #
    #####################

    def __init__(self):
        super().__init__("wheat")
        self.projected_yield_column = f"projected_{self.resource}_yield"
        self.previously_dry_column = "previous_day_dry"
        self.cumulative_dry_days_column = "cumulative_dry_days"
        self.rainfall_mid_growth_column = "rainfall_mid_growth"
        self.rainfall_late_growth_column = "rainfall_late_growth"

    def setup(self, builder: Builder) -> None:
        super().setup(builder)
        self.clock = builder.time.clock()
        self.step_size = builder.time.step_size()

        start_date = get_time_stamp(builder.configuration.time.start)
        self.next_sowing_date = get_next_annual_event_date(
            start_date,
            self.configuration.sowing_date.month,
            self.configuration.sowing_date.day,
        )
        self.next_harvest_date = get_next_annual_event_date(
            start_date,
            self.configuration.harvest_date.month,
            self.configuration.harvest_date.day,
        )

        self.effect_of_temperature_on_yield = builder.lookup.build_table(
            pd.read_csv(EFFECT_OF_TEMPERATURE_ON_WHEAT_YIELD),
            parameter_columns=["temperature"],
        )

        self.get_total_population = builder.value.get_value("total_population")

    def on_initialize_simulants(self, pop_data: SimulantData) -> None:
        super().on_initialize_simulants(pop_data)

        feature = self.population_view.subview([FEATURE]).get(pop_data.index)
        village_index = feature[feature == "village"].index
        created_columns = pd.DataFrame(
            {
                self.projected_yield_column: 0.0,
                self.previously_dry_column: False,
                self.cumulative_dry_days_column: 0.0,
                self.rainfall_mid_growth_column: 0.0,
                self.rainfall_late_growth_column: 0.0,
            },
            index=pop_data.index,
        )
        created_columns.loc[village_index, self.projected_yield_column] = (
            self.initial_village_size * self.configuration.annual_per_capita_accumulation.loc
        )
        self.population_view.update(created_columns)

    def on_time_step(self, event: Event) -> None:
        """
        Modify the expected wheat yield based on the current temperature.
        Track rainfall metrics.
        """
        # todo need to model effects of rainfall metrics on yield
        # if next harvest date is less than the next sowing date, update projected yield
        if self.next_harvest_date < self.next_sowing_date:
            data = self.population_view.get(event.index)
            effect_of_temperature = self.effect_of_temperature_on_yield(data.index)
            data[self.projected_yield_column] *= effect_of_temperature

            updates = pd.DataFrame(data[self.projected_yield_column])

            clock_time = self.clock()
            if clock_time > self.next_harvest_date - pd.Timedelta(days=90):
                updates[self.rainfall_late_growth_column] = (
                    data[self.rainfall_late_growth_column] + data["rainfall"]
                )
            elif clock_time > self.next_harvest_date - pd.Timedelta(days=180):
                updates[self.rainfall_mid_growth_column] = (
                    data[self.rainfall_mid_growth_column] + data["rainfall"]
                )

            cumulatively_dry = data[self.previously_dry_column] & (data["rainfall"] == 0.0)
            updates[self.cumulative_dry_days_column] = (
                data[self.cumulative_dry_days_column] + cumulatively_dry
            )

            self.population_view.update(updates)

        # Update stores
        super().on_time_step(event)

    def on_time_step_cleanup(self, event: Event) -> None:
        """
        Sets the previously dry day column. Updates the next sowing date if
        sowing occurred during this time-step. Resets the all yield related
        columns and updates the next harvest date if the harvest occurred during
        this time-step.
        """

        rainfall = self.population_view.get(event.index)["rainfall"]
        updates = pd.DataFrame({self.previously_dry_column: rainfall == 0.0})

        clock_time = self.clock()
        if clock_time < self.next_sowing_date <= clock_time + self.step_size():
            self.next_sowing_date += ONE_YEAR

        if clock_time < self.next_harvest_date <= clock_time + self.step_size():
            self.next_harvest_date += ONE_YEAR

            expected_yield = self.population_view.get(event.index)[
                self.projected_yield_column
            ]
            updates[self.projected_yield_column] = (
                self.total_population(expected_yield.index)
                * self.configuration.annual_per_capita_accumulation.loc
            )
            updates[
                [
                    self.cumulative_dry_days_column,
                    self.rainfall_mid_growth_column,
                    self.rainfall_late_growth_column,
                ]
            ] = 0.0

        self.population_view.update(updates)

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
        if clock_time < self.next_harvest_date <= clock_time + self.step_size():
            harvest_quantity = self.population_view.get(index)[self.projected_yield_column]
        else:
            harvest_quantity = pd.Series(
                0.0, index=index, name=f"{self.resource}.accumulation"
            )

        return harvest_quantity
