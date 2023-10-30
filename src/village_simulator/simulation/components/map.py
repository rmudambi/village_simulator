import itertools
from typing import List

import pandas as pd
from vivarium import Component
from vivarium.framework.engine import Builder
from vivarium.framework.population import SimulantData

from village_simulator.constants import Columns


class Map(Component):
    """A component that creates and manages the map"""

    CONFIGURATION_DEFAULTS = {
        "map": {
            # fixme this must have the same area as the population size
            "dimensions": {Columns.X: 8, Columns.Y: 5}
        }
    }

    ##############
    # Properties #
    ##############

    @property
    def columns_created(self) -> List[str]:
        return [Columns.X, Columns.Y, Columns.TERRAIN]

    #####################
    # Lifecycle methods #
    #####################

    def setup(self, builder: Builder) -> None:
        self.configuration = builder.configuration.map
        self.randomness = builder.randomness.get_stream(self.name)
        self.key_columns = builder.configuration.randomness.key_columns
        self.register_simulants = builder.randomness.register_simulants

    def on_initialize_simulants(self, pop_data: SimulantData) -> None:
        width = self.configuration.dimensions.x
        height = self.configuration.dimensions.y
        coordinates = pd.DataFrame(
            itertools.product(range(width), range(height)), columns=[Columns.X, Columns.Y]
        )

        self.register_simulants(coordinates[self.key_columns])
        self.population_view.update(coordinates)

        terrain = self.randomness.choice(
            pop_data.index,
            ["grassland", "desert", "forest", "mountain"],
            additional_key="initialize_terrain",
        ).rename(Columns.TERRAIN)
        self.population_view.update(terrain)
