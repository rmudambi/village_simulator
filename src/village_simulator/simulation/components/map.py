import itertools
from typing import List

import pandas as pd
from vivarium import Component
from vivarium.framework.engine import Builder
from vivarium.framework.population import SimulantData

from village_simulator.constants import Columns, paths

TERRAIN_MAPPER = {
    "D": "desert",
    "F": "forest",
    "G": "grassland",
    "M": "mountain",
}


class Map(Component):
    """A component that creates and manages the map"""

    CONFIGURATION_DEFAULTS = {
        "map": {
            # fixme this must have the same area as the population size
            #  this can be fixed by plugging into the population manager's simulant creator
            "dimensions": {Columns.X: None, Columns.Y: None},
            "terrain_source": paths.TERRAIN_MAP,
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
        terrain_source = self.configuration.terrain_source
        if terrain_source != "random":
            terrain_grid = pd.read_csv(terrain_source, header=None)
            terrain_grid.columns.name = Columns.X
            terrain_grid.index.name = Columns.Y
            maps = (
                terrain_grid.T.stack()
                .map(TERRAIN_MAPPER)
                .rename(Columns.TERRAIN)
                .reset_index()
            )
            self.register_simulants(maps[self.key_columns])
            self.population_view.update(maps)
        else:
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
