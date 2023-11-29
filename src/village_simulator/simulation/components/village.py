from typing import Dict, List

import numpy as np
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

    CONFIGURATION_DEFAULTS = {"village": {"probability": 0.2}}

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

        self.terrain_village_weight = builder.lookup.build_table(
            pd.read_csv(EFFECT_OF_TERRAIN_ON_VILLAGE), key_columns=[Columns.TERRAIN]
        )

        self.effect_of_terrain_on_arable_land = builder.lookup.build_table(
            pd.read_csv(EFFECT_OF_TERRAIN_ON_ARABLE_LAND), key_columns=[Columns.TERRAIN]
        )

    def on_initialize_simulants(self, pop_data: SimulantData) -> None:
        mean_village_probability = self.configuration.probability
        raw_terrain_weights = self.terrain_village_weight(pop_data.index)

        village_probabilities = scale_probabilities_to_mean_probability(
            mean_village_probability, raw_terrain_weights
        )

        # todo: should iteratively cap at 1.0 (and maybe warn?) rather than throwing an error
        if np.any(village_probabilities > 1.0):
            raise ValueError(
                f"Village probabilities must be <= 1.0, but found {village_probabilities}"
            )

        probabilities = pd.DataFrame(
            {True: village_probabilities, False: 1 - village_probabilities}
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
        ).clip(lower=0.0, upper=1.0)

        self.population_view.update(initial_values)


def scale_probabilities_to_mean_probability(
    target_probability: float, raw_probabilities: pd.Series
) -> pd.Series:
    """
    Scale the input probabilities so that their mean is equal to the target probability.

    Throws a ValueError if any of the scaled probabilities are greater than 1.0.
    """
    mean_weight = raw_probabilities.mean()
    probabilities = raw_probabilities * target_probability / mean_weight

    if np.any(probabilities > 1.0):
        raise ValueError(f"Probabilities must be <= 1.0, but found {probabilities}")

    return probabilities
