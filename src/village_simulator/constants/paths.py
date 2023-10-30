from pathlib import Path

import village_simulator

BASE_DIR = Path(village_simulator.__file__).resolve().parent
SIMULATION_DIR = BASE_DIR / "simulation"

GAME_SPECIFICATION = SIMULATION_DIR / "specification.yaml"

DATA_DIR = SIMULATION_DIR / "data"
EFFECT_OF_TEMPERATURE_ON_WHEAT_YIELD = DATA_DIR / "effect_of_temperature_on_wheat_yields.csv"
EFFECT_OF_TERRAIN_ON_ARABLE_LAND = DATA_DIR / "effect_of_terrain_on_arable_land.csv"
EFFECT_OF_TERRAIN_ON_VILLAGE = DATA_DIR / "effect_of_terrain_on_village.csv"
