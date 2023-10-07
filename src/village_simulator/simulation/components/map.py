import itertools
from typing import List

import pandas as pd
from vivarium import Component
from vivarium.framework.engine import Builder
from vivarium.framework.population import SimulantData

X = "x"
Y = "y"
FEATURE = "feature"


class Map(Component):
    """A component that creates and manages the map"""

    CONFIGURATION_DEFAULTS = {
        "map": {
            # fixme this must have the same area as the population size
            "dimensions": {"x": 8, "y": 5}
        }
    }

    ##############
    # Properties #
    ##############

    @property
    def columns_created(self) -> List[str]:
        return [X, Y, FEATURE]

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
            itertools.product(range(width), range(height)), columns=[X, Y]
        )

        self.register_simulants(coordinates[self.key_columns])
        self.population_view.update(coordinates)

        feature = self.randomness.choice(
            pop_data.index, ["village", "forest"], [0.1, 0.9], "initialize_features"
        ).rename(FEATURE)
        self.population_view.update(feature)