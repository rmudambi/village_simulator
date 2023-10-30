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

from village_simulator.constants import Columns, Pipelines
from village_simulator.constants.paths import EFFECT_OF_TEMPERATURE_ON_WHEAT_YIELD
from village_simulator.simulation.components.resources import Resource
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
        return super().columns_created + [Columns.PROJECTED_WHEAT_HARVEST]

    @property
    def columns_required(self) -> Optional[List[str]]:
        return super().columns_required + [Columns.ARABLE_LAND]

    @property
    def initialization_requirements(self) -> Dict[str, List[str]]:
        requirements = super().initialization_requirements
        requirements["requires_columns"] += [Columns.ARABLE_LAND]
        return requirements

    #####################
    # Lifecycle methods #
    #####################

    def __init__(self):
        super().__init__("wheat")

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

        self.get_projected_yield = builder.value.register_value_producer(
            Pipelines.PROJECTED_WHEAT_HARVEST,
            self.projected_yield_source,
            requires_columns=[Columns.PROJECTED_WHEAT_HARVEST],
        )

        self.get_total_population = builder.value.get_value(Pipelines.TOTAL_POPULATION)

        builder.value.register_value_modifier(
            Pipelines.MORTALITY_RATE,
            self.modify_mortality_rate,
            requires_values=[Pipelines.TOTAL_POPULATION, self.get_consumption.name],
            requires_streams=[self.name],
        )

    def on_initialize_simulants(self, pop_data: SimulantData) -> None:
        super().on_initialize_simulants(pop_data)

        arable_land = self.population_view.subview([Columns.ARABLE_LAND]).get(pop_data.index)

        init_data = pd.DataFrame(
            0.0, columns=[Columns.PROJECTED_WHEAT_HARVEST], index=pop_data.index
        )
        init_data.loc[
            arable_land.index, Columns.PROJECTED_WHEAT_HARVEST
        ] = self.initialize_projected_yield(arable_land.squeeze(axis=1))

        self.population_view.update(init_data)

    def on_time_step(self, event: Event) -> None:
        """
        Modify the projected wheat yield and update quantity of stored wheat.
        """
        # if next harvest date is less than the next sowing date, update projected yield
        if self.next_harvest_date < self.next_sowing_date:
            index = self.population_view.get(event.index).index
            projected_yield = self.get_projected_yield(index).rename(
                Columns.PROJECTED_WHEAT_HARVEST
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

            arable_land = self.population_view.get(event.index)[Columns.ARABLE_LAND]
            projected_yield = self.initialize_projected_yield(arable_land)
            self.population_view.update(projected_yield)

    #################
    # Setup methods #
    #################

    def register_accumulation(self, builder):
        return builder.value.register_value_producer(
            self._accumulation_pipeline,
            self.harvest_quantity_source,
            requires_values=[Pipelines.TOTAL_POPULATION],
            requires_streams=[self.name],
        )

    def register_consumption(self, builder):
        return builder.value.register_value_producer(
            self._consumption_pipeline,
            self.consumption_rate_source,
            requires_values=[Pipelines.TOTAL_POPULATION],
            requires_streams=[self.name],
        )

    ####################
    # Pipeline sources #
    ####################

    def harvest_quantity_source(self, index: pd.Index) -> pd.Series:
        """
        Gets the total amount of food accumulated during this time-step.

        Right now all food is harvested at a single time in the year

        :param index:
        :return:
        """
        clock_time = self.clock()
        if clock_time < self.next_harvest_date <= clock_time + self.step_size():
            harvest_quantity = self.population_view.get(index)[Columns.PROJECTED_WHEAT_HARVEST]
        else:
            harvest_quantity = pd.Series(0.0, index=index)

        return harvest_quantity

    def consumption_rate_source(self, index: pd.Index) -> pd.Series:
        """
        Gets the rate at which wheat is consumed by each village.

        This is an annual rate, which will be rescaled to the time-step by the
        pipeline's post-processor.

        :param index:
        :return:
        """
        wheat_stores = self.population_view.get(index)[self._stores_column]
        steps_to_next_harvest = (self.next_harvest_date - self.clock()) / self.step_size()
        store_depletion_rate = wheat_stores / steps_to_next_harvest

        natural_consumption_rate = self.get_natural_consumption_rate(index)

        consumption_rate = pd.concat(
            [wheat_stores, store_depletion_rate, natural_consumption_rate], axis=1
        ).min(axis=1)
        return consumption_rate

    def projected_yield_source(self, index: pd.Index) -> pd.Series:
        """Gets the projected yield of wheat for each village."""
        return self.population_view.get(index)[Columns.PROJECTED_WHEAT_HARVEST]

    ######################
    # Pipeline modifiers #
    ######################

    def modify_mortality_rate(self, index: pd.Index, target: pd.Series) -> pd.Series:
        """
        Gets the effect of wheat consumption on mortality rate.

        :param index:
        :param target:
        :return:
        """
        natural_consumption = self.get_natural_consumption_rate(index)
        actual_consumption = self.get_consumption(index)
        effect = (natural_consumption / actual_consumption) ** 10
        mortality_rate = target.mul(effect, axis=0)

        return mortality_rate

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
        return projected_yield.rename(Columns.PROJECTED_WHEAT_HARVEST)

    def get_natural_consumption_rate(self, index: pd.Index) -> pd.Series:
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
        return natural_consumption_rate


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

    def setup(self, builder: Builder) -> None:
        self.effect_of_temperature_on_yield = builder.lookup.build_table(
            self._get_effect_data(), parameter_columns=[Columns.TEMPERATURE]
        )

        builder.value.register_value_modifier(
            Pipelines.PROJECTED_WHEAT_HARVEST, self.modify_yield, requires_columns=[Columns.TEMPERATURE]
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
            Columns.PREVIOUSLY_DRY,
            Columns.CUMULATIVE_DRY_DAYS,
            Columns.RAINFALL_MID_GROWTH,
            Columns.RAINFALL_LATE_GROWTH,
        ]

    @property
    def columns_required(self) -> Optional[List[str]]:
        return [Columns.RAINFALL]

    @property
    def population_view_query(self) -> Optional[str]:
        return f"{Columns.IS_VILLAGE} == True"

    #####################
    # Lifecycle methods #
    #####################

    def __init__(self):
        super().__init__()
        self.target = "wheat"
        self._cumulative_rainfall_columns = [
            Columns.CUMULATIVE_DRY_DAYS,
            Columns.RAINFALL_MID_GROWTH,
            Columns.RAINFALL_LATE_GROWTH,
        ]

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
            Pipelines.PROJECTED_WHEAT_HARVEST, self.modify_yield, requires_columns=[Columns.TEMPERATURE]
        )

    def on_initialize_simulants(self, pop_data: SimulantData) -> None:
        initialization_data = pd.DataFrame(
            0.0, index=pop_data.index, columns=self._cumulative_rainfall_columns
        )
        initialization_data[Columns.PREVIOUSLY_DRY] = False
        self.population_view.update(initialization_data)

    def on_time_step_prepare(self, event: Event) -> None:
        """Track rainfall metrics that affect wheat"""
        # if next harvest date is less than the next sowing date, update projected yield
        if self.next_harvest_date < self.next_sowing_date:
            data = self.population_view.get(event.index)

            cumulatively_dry = data[Columns.PREVIOUSLY_DRY] & (data[Columns.RAINFALL] == 0.0)
            data[Columns.CUMULATIVE_DRY_DAYS] += cumulatively_dry
            updates = pd.DataFrame(data[Columns.CUMULATIVE_DRY_DAYS])

            if event.time > self.next_harvest_date - pd.Timedelta(days=90):
                updates[Columns.RAINFALL_LATE_GROWTH] = (
                    data[Columns.RAINFALL_LATE_GROWTH] + data[Columns.RAINFALL]
                )
            elif event.time > self.next_harvest_date - pd.Timedelta(days=180):
                updates[Columns.RAINFALL_MID_GROWTH] = (
                    data[Columns.RAINFALL_MID_GROWTH] + data[Columns.RAINFALL]
                )

            self.population_view.update(updates)

    def on_time_step_cleanup(self, event: Event) -> None:
        """
        Sets the previously dry day column. Updates the next sowing date if
        sowing occurred during this time-step. Resets the all yield related
        columns and updates the next harvest date if the harvest occurred during
        this time-step.
        """

        rainfall = self.population_view.get(event.index)[Columns.RAINFALL]
        updates = pd.DataFrame({Columns.PREVIOUSLY_DRY: rainfall == 0.0})

        # todo get this information from the wheat component
        if event.time - self.step_size() < self.next_sowing_date <= event.time:
            self.next_sowing_date += ONE_YEAR

        if event.time - self.step_size() < self.next_harvest_date <= event.time:
            # todo get this information from the wheat component
            self.next_harvest_date += ONE_YEAR
            updates[self._cumulative_rainfall_columns] = 0.0

        self.population_view.update(updates)

    ######################
    # Pipeline modifiers #
    ######################

    def modify_yield(self, index: pd.Index, target: pd.Series) -> pd.Series:
        """Gets the effect of rainfall on wheat yield."""
        clock_time = self.clock()
        event_time = self.clock() + self.step_size()

        if clock_time < self.next_harvest_date - pd.Timedelta(days=90) <= event_time:
            rainfall = self.population_view.get(index)[Columns.RAINFALL_MID_GROWTH]
            effect = self.mid_growth.get_effect(rainfall)

        elif clock_time < self.next_harvest_date <= event_time:
            data = self.population_view.get(index)[
                [Columns.CUMULATIVE_DRY_DAYS, Columns.RAINFALL_LATE_GROWTH]
            ]
            cumulative_dry_days_effect = self.cumulative_dry_days.get_effect(
                data[Columns.CUMULATIVE_DRY_DAYS]
            )
            rainfall_late_growth_effect = self.late_growth.get_effect(
                data[Columns.RAINFALL_LATE_GROWTH]
            )
            effect = cumulative_dry_days_effect * rainfall_late_growth_effect
        else:
            return target

        target = target * effect
        return target
