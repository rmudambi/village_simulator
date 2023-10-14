import numpy as np
import pandas as pd
import pytest

from village_simulator.simulation.distributions import (  # zero_inflated_gamma_ppf,
    stretched_truncnorm_ppf,
)


@pytest.mark.parametrize(
    "ppf, kwargs",
    [
        (stretched_truncnorm_ppf, {"loc": 4.0, "scale": 1.0}),
        # (zero_inflated_gamma_ppf, {"zero_probability": 0.6, "shape": 1.0, "scale": 1.0}),
    ],
)
def test_ppf_all_scalar_return_float(ppf, kwargs):
    assert isinstance(ppf(0.5, **kwargs), float)


@pytest.mark.parametrize(
    "ppf, kwargs",
    [
        (stretched_truncnorm_ppf, {"loc": pd.Series([4.0, 6.0]), "scale": 1.0}),
        (stretched_truncnorm_ppf, {"loc": 4.0, "scale": pd.Series([1.0, 2.0])}),
        # (zero_inflated_gamma_ppf, {"zero_probability": 0.6, "shape": 1.0, "scale": 1.0}),
    ],
)
def test_ppf_quantiles_scalar_kwargs_series_or_scalar_returns_series(ppf, kwargs):
    # test that the function returns a Series when quantiles is a scalar and the
    # distribution kwargs are a mix of Series and scalars with at least one
    # Series
    assert isinstance(ppf(0.5, **kwargs), pd.Series)


@pytest.mark.parametrize(
    "ppf, kwargs",
    [
        (stretched_truncnorm_ppf, {"loc": pd.Series([4.0, 6.0]), "scale": 1.0}),
        (stretched_truncnorm_ppf, {"loc": 4.0, "scale": pd.Series([1.0, 2.0])}),
        # (zero_inflated_gamma_ppf, {"zero_probability": 0.6, "shape": 1.0, "scale": 1.0}),
    ],
)
def test_ppf_quantiles_series_kwargs_series_or_scalar_returns_dataframe(ppf, kwargs):
    # test that the function returns a DataFrame when quantiles is a Series and
    # the distribution kwargs are a mix of Series and scalars with at least one
    # Series
    assert isinstance(ppf(pd.Series([0.5, 0.99]), **kwargs), pd.DataFrame)


@pytest.mark.parametrize(
    "ppf, kwargs",
    [
        (
            stretched_truncnorm_ppf,
            {"loc": pd.Series([4.0, 6.0]), "scale": pd.Series([1.0, 2.0])},
        ),
        # (zero_inflated_gamma_ppf, {"zero_probability": 0.6, "shape": 1.0, "scale": 1.0}),
    ],
)
def test_ppf_all_series_returns_dataframe(ppf, kwargs):
    # test that the function returns a dataframe when quantiles and all kwargs
    # are Series, even if quantiles has a different index
    assert isinstance(ppf(pd.Series([0.5, 0.99, 0.63]), **kwargs), pd.DataFrame)


@pytest.mark.parametrize(
    "ppf, kwargs",
    [
        (
            stretched_truncnorm_ppf,
            {"loc": pd.Series([4.0, 6.0, 2.0]), "scale": pd.Series([1.0, 2.0])},
        ),
        # (zero_inflated_gamma_ppf, {"zero_probability": 0.6, "shape": 1.0, "scale": 1.0}),
    ],
)
def test_ppf_mismatched_distribution_parameters(ppf, kwargs):
    # test that the function raises an error when kwargs have different indices
    with pytest.raises(
        ValueError,
        match="All distribution parameters must have the same index or be scalars.",
    ):
        ppf(pd.Series([0.5, 0.99, 0.63, 0.01]), **kwargs)


def test_stretched_truncnorm_ppf_loc_zero_all_scalar():
    # Test that the function returns 0 when loc is 0
    assert stretched_truncnorm_ppf(0.5, 0.0, 1.0) == 0.0


def test_stretched_truncnorm_ppf_loc_zero_quantiles_series():
    # Test that the function returns 0 when loc is 0
    quantiles = pd.Series([0.01, 0.3, 0.89, 0.99])
    assert stretched_truncnorm_ppf(quantiles, 0.0, 1.0).eq(0.0).all()


def test_stretched_truncnorm_ppf_loc_some_zero():
    # test that the function returns a dataframe with 0 for all values in
    # columns where loc is 0
    quantiles = pd.Series([0.99, 0.25, 0.01])
    loc = pd.Series([5.4, 0.0, 0.0, 4.1, 0.0])
    scale = pd.Series(np.random.rand(len(loc)))

    actual = stretched_truncnorm_ppf(quantiles, loc, scale)

    assert actual.loc[:, loc[loc == 0.0].index].eq(0.0).all().all()
