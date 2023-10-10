import pytest

from village_simulator.simulation import Demographics


def test_get_fertility_rate(mocker):
    # todo
    demographics = Demographics()
    demographics.configuration = Demographics.CONFIGURATION_DEFAULTS["demographics"][
        "fertility_rate"
    ]

    # todo mock frozen_dist.sample to give different numbers for male and female

    ...


def test_get_mortality_rate():
    # todo
    ...


def test_get_total_population():
    # todo
    ...


def test_on_time_step():
    # todo
    ...


def test_initialization():
    # todo
    ...
