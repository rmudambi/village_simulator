import pandas as pd
import pytest
from scipy import stats
from vivarium import ConfigTree

from village_simulator.simulation.distributions import (
    NORMAL,
    FrozenDistribution,
    _stretched_truncnorm_ppf,
    _zero_inflated_gamma_ppf,
)


@pytest.fixture()
def quantiles() -> pd.Series:
    return pd.Series([0.53, 0.21, 0.42, 0.45, 0.04, 0.74, 0.66, 0.92, 0.23, 0.77])


@pytest.mark.parametrize("is_config_tree", [True, False])
def test_sample_with_index(mocker, quantiles, is_config_tree):
    loc, scale = (1.0, 2.0)

    randomness_stream = mocker.Mock()
    randomness_stream.get_draw = lambda idx, key: quantiles[idx]
    expected = stats.norm.ppf(quantiles, loc=loc, scale=scale)

    config = {"loc": loc, "scale": scale}
    if is_config_tree:
        config = ConfigTree(config)

    dist = FrozenDistribution(NORMAL, config)
    actual = dist.sample(randomness_stream, "some_key", quantiles.index)

    assert (expected == actual).all()


@pytest.mark.parametrize("is_config_tree", [True, False])
def test_sample_with_no_index(mocker, is_config_tree):
    quantile, loc, scale = (0.4, 1.0, 2.0)

    randomness_stream = mocker.Mock()
    randomness_stream.get_draw = lambda idx, key: pd.Series([quantile])
    expected = stats.norm.ppf(quantile, loc=loc, scale=scale)

    config = {"loc": loc, "scale": scale}
    if is_config_tree:
        config = ConfigTree(config)

    dist = FrozenDistribution(NORMAL, config)
    actual = dist.sample(randomness_stream, "some_key", None)

    assert expected == actual


@pytest.mark.parametrize("is_config_tree", [True, False])
def test_sample_from_nested_distribution(mocker, quantiles, is_config_tree):
    # todo
    pass
