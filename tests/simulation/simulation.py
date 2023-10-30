from vivarium import InteractiveContext

from village_simulator.constants import paths


def test_simulation():
    sim = InteractiveContext(paths.GAME_SPECIFICATION)
    for i in range(400):
        sim.step()
