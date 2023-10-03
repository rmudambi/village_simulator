from pathlib import Path

import village_simulator


BASE_DIR = Path(village_simulator.__file__).resolve().parent
GAME_SPECIFICATION = BASE_DIR / "simulation" / "specification.yaml"

