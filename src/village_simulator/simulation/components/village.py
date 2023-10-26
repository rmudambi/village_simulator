from typing import List, Dict

import pandas as pd
from vivarium import Component
from vivarium.framework.engine import Builder
from vivarium.framework.population import SimulantData

from village_simulator.paths import EFFECT_OF_TERRAIN_ON_VILLAGE
from village_simulator.simulation.components.map import TERRAIN

IS_VILLAGE = "is_village"

class Village(Component):
    """
    A component that creates and manages physical features relevant to villages
    """

    CONFIGURATION_DEFAULTS = {
        "village": {
            "probability": 0.4
        }
    }

    ##############
    # Properties #
    ##############

    @property
    def columns_created(self) -> List[str]:
        return [IS_VILLAGE]

    @property
    def initialization_requirements(self) -> Dict[str, List[str]]:
        return {"requires_columns": [TERRAIN], "requires_streams": [self.name]}

    #####################
    # Lifecycle methods #
    #####################

    def setup(self, builder: Builder) -> None:
        self.configuration = builder.configuration.village
        self.randomness = builder.randomness.get_stream(self.name)

        self.effect_of_terrain_on_village = builder.lookup.build_table(
            pd.read_csv(EFFECT_OF_TERRAIN_ON_VILLAGE), key_columns=[TERRAIN]
        )

    def on_initialize_simulants(self, pop_data: SimulantData) -> None:
        village_probability = self.effect_of_terrain_on_village(pop_data.index)
        probabilities = pd.DataFrame({True: village_probability, False: 1 - village_probability})

        is_village = self.randomness.choice(
            pop_data.index,
            [True, False],
            probabilities.to_numpy(),
            "initialize_village",
        ).rename(IS_VILLAGE)
        self.population_view.update(is_village)
