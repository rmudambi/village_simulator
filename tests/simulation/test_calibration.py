from village_simulator.simulation.calibration.rainfall import sample_mid_growth_rainfall


def test_sample_mid_growth_rainfall():
    sample_mid_growth_rainfall(years=100, tiles=40)
