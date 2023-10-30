from typing import Dict, List

import pandas as pd
from scipy import stats
from vivarium import Component
from vivarium.framework.engine import Builder
from vivarium.framework.population import SimulantData

from village_simulator.constants import Columns
from village_simulator.constants.paths import (
    EFFECT_OF_TERRAIN_ON_ARABLE_LAND,
    EFFECT_OF_TERRAIN_ON_VILLAGE,
)


class Village(Component):
    """
    A component that creates and manages physical features relevant to villages
    """

    CONFIGURATION_DEFAULTS = {"village": {"probability": 0.4}}

    ##############
    # Properties #
    ##############

    @property
    def columns_created(self) -> List[str]:
        return [Columns.IS_VILLAGE, Columns.ARABLE_LAND]

    @property
    def initialization_requirements(self) -> Dict[str, List[str]]:
        return {"requires_columns": [Columns.TERRAIN], "requires_streams": [self.name]}

    #####################
    # Lifecycle methods #
    #####################

    def setup(self, builder: Builder) -> None:
        self.configuration = builder.configuration.village
        self.randomness = builder.randomness.get_stream(self.name)

        self.effect_of_terrain_on_village = builder.lookup.build_table(
            pd.read_csv(EFFECT_OF_TERRAIN_ON_VILLAGE), key_columns=[Columns.TERRAIN]
        )

        self.effect_of_terrain_on_arable_land = builder.lookup.build_table(
            pd.read_csv(EFFECT_OF_TERRAIN_ON_ARABLE_LAND), key_columns=[Columns.TERRAIN]
        )

    def on_initialize_simulants(self, pop_data: SimulantData) -> None:
        village_probability = self.effect_of_terrain_on_village(pop_data.index)
        probabilities = pd.DataFrame(
            {True: village_probability, False: 1 - village_probability}
        )

        initial_values = pd.DataFrame(index=pop_data.index)

        initial_values[Columns.IS_VILLAGE] = self.randomness.choice(
            pop_data.index,
            [True, False],
            probabilities.to_numpy(),
            "initialize_village",
        )

        arable_land_data = self.effect_of_terrain_on_arable_land(pop_data.index)
        initial_values[Columns.ARABLE_LAND] = self.randomness.sample_from_distribution(
            pop_data.index,
            stats.norm,
            additional_key="arable_land",
            loc=arable_land_data["loc"],
            scale=arable_land_data["scale"],
        )

        self.population_view.update(initial_values)
