from typing import Dict, Union

import numpy as np
import pandas as pd
import pytest

from village_simulator.simulation.distributions import (
    stretched_truncnorm_ppf,
    zero_inflated_gamma_ppf,
)


def _kwargs_list_to_series(
    kwargs: Dict[str, Union[float, list]]
) -> Dict[str, Union[float, pd.Series]]:
    """
    Convert a dictionary of kwargs to a dictionary of kwargs where all values
    that were previously lists are now Series.
    """
    return {
        key: pd.Series(value) if isinstance(value, list) else value
        for key, value in kwargs.items()
    }


SCALAR_DIST_PARAMS = [
    (stretched_truncnorm_ppf, {"loc": 4.0, "scale": 1.0}),
    (zero_inflated_gamma_ppf, {"p_zero": 0.6, "shape": 1.0, "scale": 1.0}),
]


@pytest.mark.parametrize("ppf, kwargs", SCALAR_DIST_PARAMS)
def test_ppf_all_scalar_return_float(ppf, kwargs):
    assert isinstance(ppf(0.5, **kwargs), float)


@pytest.mark.parametrize("ppf, kwargs", SCALAR_DIST_PARAMS)
def test_ppf_quantiles_series_kwargs_scalar_returns_series(ppf, kwargs):
    """
    Test that the function returns a Series when quantiles is a Series and the
    distribution kwargs are all scalars
    """
    actual = ppf(pd.Series([0.5, 0.99, 0.01]), **kwargs)

    assert isinstance(actual, pd.Series)
    assert len(actual) == 3


@pytest.mark.parametrize(
    "ppf, kwargs",
    [
        (
            zero_inflated_gamma_ppf,
            {
                "p_zero": np.array([[0.3], [0.1]]),
                "shape": 1.0,
                "scale": np.array([[1.0], [3.0]]),
            },
        ),
        (zero_inflated_gamma_ppf, {"p_zero": 0.1, "shape": 1.0, "scale": 3.0}),
    ],
)
def test_ppf_quantiles_series_returns_same_dimension_series(ppf, kwargs):
    """
    Test that the function returns a Series when quantiles is a Series and the
    distribution kwargs are all ndarrays with the same shape or scalars.
    """
    quantiles = pd.Series([0.5, 0.01])
    actual = zero_inflated_gamma_ppf(quantiles, **kwargs)

    assert isinstance(actual, pd.Series)
    assert len(actual) == len(quantiles)


@pytest.mark.parametrize(
    "ppf, kwargs",
    [
        (
            zero_inflated_gamma_ppf,
            {
                "p_zero": np.array([[0.3, 0.1, 0.9]]),
                "shape": 1.0,
                "scale": np.array([[1.0, 3.0, 2.0]]),
            },
        )
    ],
)
def test_ppf_quantiles_series_kwargs_same_dimension_series(ppf, kwargs):
    """
    Test that the function returns a Series when quantiles is a Series and the
    distribution kwargs are all ndarrays with the same shape.
    """
    quantiles = pd.Series([0.5, 0.01])
    actual = zero_inflated_gamma_ppf(quantiles, **kwargs)

    assert isinstance(actual, pd.DataFrame)
    assert actual.shape == (2, 3)


# MIXED_DIST_PARAMS = [
#     (stretched_truncnorm_ppf, {"loc": [4.0, 6.0], "scale": 1.0}),
#     (stretched_truncnorm_ppf, {"loc": 4.0, "scale": [1.0, 2.0]}),
#     (zero_inflated_gamma_ppf, {"p_zero": [0.3, 0.6], "shape": 1.0, "scale": 1.0}),
#     (zero_inflated_gamma_ppf, {"p_zero": 0.6, "shape": [1.0, 1.5], "scale": 1.0}),
#     (zero_inflated_gamma_ppf, {"p_zero": 0.6, "shape": 1.0, "scale": [1.0, 5.0]}),
# ]
#
#
# @pytest.mark.parametrize("ppf, kwargs", MIXED_DIST_PARAMS)
# def test_ppf_quantiles_scalar_kwargs_series_or_scalar_returns_series(ppf, kwargs):
#     """
#     Test that the function returns a Series when quantiles is a scalar and the
#     distribution kwargs are a mix of Series and scalars with at least one Series
#     """
#     kwargs = _kwargs_list_to_series(kwargs)
#     actual = ppf(0.5, **kwargs)
#
#     assert isinstance(actual, pd.Series)
#     assert len(actual) == 2


# @pytest.mark.parametrize("ppf, kwargs", MIXED_DIST_PARAMS)
# def test_ppf_quantiles_series_kwargs_series_or_scalar_returns_dataframe(ppf, kwargs):
#     """
#     Test that the function returns a DataFrame when quantiles is a Series and
#     the distribution kwargs are a mix of Series and scalars with at least one
#     Series
#     """
#     kwargs = _kwargs_list_to_series(kwargs)
#     actual = ppf(pd.Series([0.5, 0.99, 0.01]), **kwargs)
#
#     assert isinstance(actual, pd.DataFrame)
#     assert actual.shape == (3, 2)


@pytest.mark.parametrize(
    "ppf, kwargs, message",
    [
        (stretched_truncnorm_ppf, {"loc": [4.0, 6.0, 2.0], "scale": [1.0, 2.0]}, "ga"),
        (
            zero_inflated_gamma_ppf,
            {"p_zero": np.array([[0.3, 0.6]]), "shape": np.array([[1.0, 0.5, 0.6]])},
            "same shape",
        ),
        (
            zero_inflated_gamma_ppf, {"p_zero": np.array([[0.3], [0.6]])}, "same length as quantiles",
        ),
        (
            zero_inflated_gamma_ppf, {"shape": np.array([[0.3, 4.0], [0.6, 7.0]])}, "must be one-dimensional",
        ),
    ],
)
def test_ppf_mismatched_distribution_parameters(ppf, kwargs, message):
    """Test that the function raises an error when kwargs have different indices"""
    with pytest.raises(ValueError, match=message,):
        ppf(pd.Series([0.5, 0.99, 0.63, 0.01]), **kwargs)


def test_stretched_truncnorm_ppf_loc_zero_all_scalar():
    """Test that the function returns 0 when loc is 0"""
    assert stretched_truncnorm_ppf(0.5, 0.0, 1.0) == 0.0


def test_stretched_truncnorm_ppf_loc_zero_quantiles_series():
    """Test that the function returns 0 when loc is 0"""
    quantiles = pd.Series([0.01, 0.3, 0.89, 0.99])
    assert stretched_truncnorm_ppf(quantiles, 0.0, 1.0).eq(0.0).all()


def test_stretched_truncnorm_ppf_loc_some_zero():
    """
    Test that the function returns a dataframe with 0 for all values in columns
    where loc is 0
    """
    quantiles = pd.Series([0.99, 0.25, 0.01])
    loc = pd.Series([5.4, 0.0, 0.0, 4.1, 0.0])
    scale = pd.Series(np.random.rand(len(loc)))

    actual = stretched_truncnorm_ppf(quantiles, loc, scale)

    assert actual.loc[:, loc[loc == 0.0].index].eq(0.0).all().all()


def test_zero_inflated_gamma_ppf_returns_zero_scalar():
    """Test that the function returns 0 when quantile is less than p_zero"""
    assert zero_inflated_gamma_ppf(0.5, 0.5, 1.0, 1.0) == 0.0


@pytest.mark.parametrize(
    "p_zero, expected",
    [
        (0.5, pd.Series([True, False, True, False])),
        (np.array([[0.4], [0.3], [0.1], [0.7]]), pd.Series([True, False, False, False])),
    ],
)
def test_zero_inflated_gamma_ppf_returns_zero_quantiles_series(p_zero, expected):
    """Test that the function returns 0 when quantile is less than p_zero"""
    quantiles = pd.Series([0.01, 0.89, 0.3, 0.99])
    actual = zero_inflated_gamma_ppf(quantiles, p_zero, 1.0, 1.0).eq(0.0)

    pd.testing.assert_series_equal(actual, expected, check_names=False)


def test_zero_inflated_gamma_ppf_returns_zero_2d():
    """Test that the function returns 0 when quantile is less than p_zero"""
    quantiles = pd.Series([0.01, 0.89, 0.3, 0.99])
    p_zero = np.array([[0.5, 0.2, 0.9]])
    expected = pd.DataFrame(
        [
            [True, True, True],
            [False, False, True],
            [True, False, True],
            [False, False, False],
        ]
    )
    actual = zero_inflated_gamma_ppf(quantiles, p_zero, 1.0, 1.0).eq(0.0)

    pd.testing.assert_frame_equal(actual, expected, check_names=False)
