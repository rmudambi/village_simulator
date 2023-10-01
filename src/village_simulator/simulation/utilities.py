import numpy as np
import pandas as pd
from vivarium.framework.randomness import RandomnessStream


def round_stochastic(
    values: pd.Series, randomness_stream: RandomnessStream, additional_key: str = ""
) -> pd.Series:
    rounding_propensities = randomness_stream.get_draw(
        values.index, additional_key=f"{additional_key}_stochastic_rounding"
    )
    floored_values = np.floor(values)
    thresholds = values - floored_values
    values = (rounding_propensities < thresholds) + floored_values
    return values.astype(int)
