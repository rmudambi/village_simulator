from vivarium import InteractiveContext

def test_simulation():
    sim = InteractiveContext("../src/village_simulator/simulation/specification.yaml")
    sim.step()
    sim.step()