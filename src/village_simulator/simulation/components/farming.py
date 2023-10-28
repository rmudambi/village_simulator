import dataclasses
from typing import Dict, List, Optional

import pandas as pd
from scipy import stats
from vivarium import Component
from vivarium.framework.engine import Builder
from vivarium.framework.event import Event
from vivarium.framework.lookup import LookupTableData
from vivarium.framework.population import SimulantData
from vivarium.framework.time import get_time_stamp
from vivarium.framework.utilities import from_yearly

from village_simulator.paths import EFFECT_OF_TEMPERATURE_ON_WHEAT_YIELD
from village_simulator.simulation.components.resources import Resource
from village_simulator.simulation.components.village import ARABLE_LAND, IS_VILLAGE
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
            "initial_per_capita_stores": {"loc": 0.5, "scale": 0.1},
            "annual_per_capita_consumption": {"loc": 0.5, "scale": 0.025},
            "land_cultivation_per_capita": {"loc": 5e-4, "scale": 1e-5},
            "land_productivity": {"loc": 1000.0, "scale": 10.0},
            "sowing_date": WHEAT_SOWING_DATE,
            "harvest_date": WHEAT_HARVEST_DATE,
        }
    }

    ##############
    # Properties #
    ##############

    @property
    def columns_created(self) -> List[str]:
        return super().columns_created + [self.projected_yield_column]

    @property
    def columns_required(self) -> Optional[List[str]]:
        return super().columns_required + [ARABLE_LAND]

    @property
    def initialization_requirements(self) -> Dict[str, List[str]]:
        requirements = super().initialization_requirements
        requirements["requires_columns"] += [ARABLE_LAND]
        return requirements

    #####################
    # Lifecycle methods #
    #####################

    def __init__(self):
        super().__init__("wheat")
        self.projected_yield_column = f"{self.resource}_projected_yield"

    def setup(self, builder: Builder) -> None:
        super().setup(builder)
        self.clock = builder.time.clock()
        self.step_size = builder.time.step_size()

        # todo expose this data in an appropriate way
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

        self.projected_yield_pipeline = builder.value.register_value_producer(
            f"{self.resource}.projected_yield",
            self.get_projected_yield,
            requires_columns=[self.projected_yield_column],
        )

        self.effect_of_temperature_on_yield = builder.lookup.build_table(
            pd.read_csv(EFFECT_OF_TEMPERATURE_ON_WHEAT_YIELD),
            parameter_columns=["temperature"],
        )

        self.get_total_population = builder.value.get_value("total_population")

    def on_initialize_simulants(self, pop_data: SimulantData) -> None:
        super().on_initialize_simulants(pop_data)

        arable_land = self.population_view.subview([ARABLE_LAND]).get(pop_data.index)

        init_data = pd.DataFrame(
            0.0, columns=[self.projected_yield_column], index=pop_data.index
        )
        init_data.loc[
            arable_land.index, self.projected_yield_column
        ] = self.initialize_projected_yield(arable_land.squeeze(axis=1))

        self.population_view.update(init_data)

    def on_time_step(self, event: Event) -> None:
        """
        Modify the projected wheat yield and update quantity of stored wheat.
        """
        # if next harvest date is less than the next sowing date, update projected yield
        if self.next_harvest_date < self.next_sowing_date:
            index = self.population_view.get(event.index).index
            projected_yield = self.projected_yield_pipeline(index).rename(
                self.projected_yield_column
            )
            self.population_view.update(projected_yield)

        # Update stores
        super().on_time_step(event)

    def on_time_step_cleanup(self, event: Event) -> None:
        """
        Updates the next sowing date if sowing occurred during this time-step.

        Resets the projected yield column and updates the next harvest date if
        the harvest occurred during this time-step.
        """
        clock_time = self.clock()
        if clock_time < self.next_sowing_date <= clock_time + self.step_size():
            self.next_sowing_date += ONE_YEAR

        if clock_time < self.next_harvest_date <= clock_time + self.step_size():
            self.next_harvest_date += ONE_YEAR

            arable_land = self.population_view.get(event.index)[ARABLE_LAND]
            projected_yield = self.initialize_projected_yield(arable_land)
            self.population_view.update(projected_yield)

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

    def register_consumption(self, builder):
        return builder.value.register_value_producer(
            f"{self.resource}.consumption",
            self.get_consumption_rate,
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

    def get_consumption_rate(self, index: pd.Index) -> pd.Series:
        """
        Gets the rate at which wheat is consumed by each village.

        This is an annual rate, which will be rescaled to the time-step by the
        pipeline's post-processor.

        :param index:
        :return:
        """
        wheat_stores = self.population_view.get(index)[self.resource_stores]
        steps_to_next_harvest = (self.next_harvest_date - self.clock()) / self.step_size()
        store_depletion_rate = wheat_stores / steps_to_next_harvest

        natural_consumption_rate = from_yearly(
            self.total_population(index)
            * self.randomness.sample_from_distribution(
                index,
                distribution=stats.norm,
                additional_key=f"{self.resource}.consumption",
                **self.configuration.annual_per_capita_consumption.to_dict(),
            ),
            self.step_size(),
        )

        consumption_rate = pd.concat(
            [wheat_stores, store_depletion_rate, natural_consumption_rate], axis=1
        ).min(axis=1)
        return consumption_rate

    def get_projected_yield(self, index: pd.Index) -> pd.Series:
        """Gets the projected yield of wheat for each village."""
        return self.population_view.get(index)[self.projected_yield_column]

    ##################
    # Helper methods #
    ##################

    def initialize_projected_yield(self, arable_land: pd.Series) -> pd.Series:
        """
        Gets the projected yield of wheat for each village.

        :param arable_land:
        :return:
        """
        land_cultivation_per_capita = self.randomness.sample_from_distribution(
            arable_land.index,
            stats.norm,
            additional_key="land_cultivation",
            **self.configuration.land_cultivation_per_capita.to_dict(),
        )
        land_under_cultivation = arable_land.combine(
            self.total_population(arable_land.index) * land_cultivation_per_capita, min
        )

        projected_yield = land_under_cultivation * self.randomness.sample_from_distribution(
            land_under_cultivation.index,
            stats.norm,
            additional_key="projected_yield",
            **self.configuration.land_productivity.to_dict(),
        )
        return projected_yield.rename(self.projected_yield_column)


class TemperatureEffect(Component):
    """
    Component that manages the effect of temperature on wheat yield.
    """

    #####################
    # Lifecycle methods #
    #####################

    def __init__(self, target: str):
        super().__init__()
        self.target = target
        self.projected_yield_pipeline = f"{self.target}.projected_yield"

    def setup(self, builder: Builder) -> None:
        self.effect_of_temperature_on_yield = builder.lookup.build_table(
            self._get_effect_data(), parameter_columns=["temperature"]
        )

        builder.value.register_value_modifier(
            self.projected_yield_pipeline, self.modify_yield, requires_columns=["temperature"]
        )

    ######################
    # Pipeline modifiers #
    ######################

    def modify_yield(self, index: pd.Index, target: pd.Series) -> pd.Series:
        """
        Gets the effect of temperature on wheat yield.

        :param index:
        :param target:
        :return:
        """
        return target * self.effect_of_temperature_on_yield(index)

    ##################
    # Helper methods #
    ##################

    def _get_effect_data(self) -> LookupTableData:
        data_path = {"wheat": EFFECT_OF_TEMPERATURE_ON_WHEAT_YIELD}[self.target]
        data = pd.read_csv(data_path)
        return data


@dataclasses.dataclass
class AccumulationWindow:
    name: str
    start: pd.Timedelta
    end: pd.Timedelta
    exposure_mean: float
    exposure_sd: float
    effect_mean: float
    effect_sd: float
    correlation_type: str = "positive"

    def get_effect(self, exposure_data) -> pd.Series:
        z_scores = (exposure_data - self.exposure_mean) / self.exposure_sd
        if self.correlation_type == "negative":
            z_scores = -z_scores
        effects = self.effect_mean + self.effect_sd * z_scores
        return effects


class RainfallEffectOnWheat(Component):
    CONFIGURATION_DEFAULTS = {
        "effect_of_rainfall_on_wheat": {
            "cumulative_dry_days": {
                # todo
            }
        }
    }

    cumulative_dry_days = AccumulationWindow(
        name="cumulative_dry_days",
        start=pd.Timedelta(days=-180),
        end=pd.Timedelta(days=0),
        # todo the effects are reversed - need yield to be inversely proportional to dry days
        exposure_mean=90.34,
        exposure_sd=9.412,
        effect_mean=1.0,
        effect_sd=0.1,
        correlation_type="negative",
    )

    mid_growth = AccumulationWindow(
        name="rainfall_mid_growth",
        start=pd.Timedelta(days=-180),
        end=pd.Timedelta(days=-90),
        exposure_mean=854.1,
        exposure_sd=139.3,
        effect_mean=1.0,
        effect_sd=0.2,
    )

    late_growth = AccumulationWindow(
        name="rainfall_late_growth",
        start=pd.Timedelta(days=-90),
        end=pd.Timedelta(days=0),
        exposure_mean=837.5,
        exposure_sd=135.6,
        effect_mean=1.0,
        effect_sd=0.3,
    )

    ##############
    # Properties #
    ##############

    @property
    def columns_created(self) -> List[str]:
        return [
            self.previously_dry_column,
            self.cumulative_dry_days_column,
            self.rainfall_mid_growth_column,
            self.rainfall_late_growth_column,
        ]

    @property
    def columns_required(self) -> Optional[List[str]]:
        return ["rainfall"]

    @property
    def population_view_query(self) -> Optional[str]:
        return f"{IS_VILLAGE} == True"

    #####################
    # Lifecycle methods #
    #####################

    def __init__(self):
        super().__init__()
        self.target = "wheat"
        self.previously_dry_column = "previous_day_dry"
        self.cumulative_dry_days_column = "cumulative_dry_days"
        self.rainfall_mid_growth_column = "rainfall_mid_growth"
        self.rainfall_late_growth_column = "rainfall_late_growth"

        self.projected_yield_pipeline = f"{self.target}.projected_yield"

    def setup(self, builder: Builder) -> None:
        self.configuration = builder.configuration.effect_of_rainfall_on_wheat
        self.target_configuration = builder.configuration[self.target]
        self.clock = builder.time.clock()
        self.step_size = builder.time.step_size()

        # todo get this information directly from the Wheat component
        start_date = get_time_stamp(builder.configuration.time.start)
        self.next_sowing_date = get_next_annual_event_date(
            start_date,
            self.target_configuration.sowing_date.month,
            self.target_configuration.sowing_date.day,
        )
        self.next_harvest_date = get_next_annual_event_date(
            start_date,
            self.target_configuration.harvest_date.month,
            self.target_configuration.harvest_date.day,
        )

        self.effect_of_cumulative_dry_days_on_yield = builder.lookup.build_table(
            # todo provide real data
            0.9,
        )

        builder.value.register_value_modifier(
            self.projected_yield_pipeline, self.modify_yield, requires_columns=["temperature"]
        )

    def on_initialize_simulants(self, pop_data: SimulantData) -> None:
        initialization_data = {
            self.previously_dry_column: False,
            self.cumulative_dry_days_column: 0.0,
            self.rainfall_mid_growth_column: 0.0,
            self.rainfall_late_growth_column: 0.0,
        }
        created_columns = pd.DataFrame(initialization_data, index=pop_data.index)
        self.population_view.update(created_columns)

    def on_time_step_prepare(self, event: Event) -> None:
        """Track rainfall metrics that affect wheat"""
        # if next harvest date is less than the next sowing date, update projected yield
        if self.next_harvest_date < self.next_sowing_date:
            data = self.population_view.get(event.index)

            cumulatively_dry = data[self.previously_dry_column] & (data["rainfall"] == 0.0)
            data[self.cumulative_dry_days_column] += cumulatively_dry
            updates = pd.DataFrame(data[self.cumulative_dry_days_column])

            if event.time > self.next_harvest_date - pd.Timedelta(days=90):
                updates[self.rainfall_late_growth_column] = (
                    data[self.rainfall_late_growth_column] + data["rainfall"]
                )
            elif event.time > self.next_harvest_date - pd.Timedelta(days=180):
                updates[self.rainfall_mid_growth_column] = (
                    data[self.rainfall_mid_growth_column] + data["rainfall"]
                )

            self.population_view.update(updates)

    def on_time_step_cleanup(self, event: Event) -> None:
        """
        Sets the previously dry day column. Updates the next sowing date if
        sowing occurred during this time-step. Resets the all yield related
        columns and updates the next harvest date if the harvest occurred during
        this time-step.
        """

        rainfall = self.population_view.get(event.index)["rainfall"]
        updates = pd.DataFrame({self.previously_dry_column: rainfall == 0.0})

        # todo get this information from the wheat component
        if event.time - self.step_size() < self.next_sowing_date <= event.time:
            self.next_sowing_date += ONE_YEAR

        if event.time - self.step_size() < self.next_harvest_date <= event.time:
            # todo get this information from the wheat component
            self.next_harvest_date += ONE_YEAR
            updates[
                [
                    self.cumulative_dry_days_column,
                    self.rainfall_mid_growth_column,
                    self.rainfall_late_growth_column,
                ]
            ] = 0.0

        self.population_view.update(updates)

    ######################
    # Pipeline modifiers #
    ######################

    def modify_yield(self, index: pd.Index, target: pd.Series) -> pd.Series:
        """Gets the effect of rainfall on wheat yield."""
        clock_time = self.clock()
        event_time = self.clock() + self.step_size()

        if clock_time < self.next_harvest_date - pd.Timedelta(days=90) <= event_time:
            rainfall = self.population_view.get(index)[self.rainfall_mid_growth_column]
            effect = self.mid_growth.get_effect(rainfall)

        elif clock_time < self.next_harvest_date <= event_time:
            data = self.population_view.get(index)[
                [self.cumulative_dry_days_column, self.rainfall_late_growth_column]
            ]
            cumulative_dry_days_effect = self.cumulative_dry_days.get_effect(
                data[self.cumulative_dry_days_column]
            )
            rainfall_late_growth_effect = self.late_growth.get_effect(
                data[self.rainfall_late_growth_column]
            )
            effect = cumulative_dry_days_effect * rainfall_late_growth_effect
        else:
            return target

        target = target * effect
        return target
