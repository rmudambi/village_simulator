from typing import Any

import numpy as np
from vivarium import InteractiveContext

from village_simulator.interface.utilities import make_bold
from village_simulator.simulation.components.map import X, Y
from village_simulator.simulation.components.village import IS_VILLAGE


def step(simulation: InteractiveContext, **_: Any) -> str:
    """Progress the game."""
    output = f"\nAdvancing the game by {simulation.configuration.time.step_size} days.\n\n"
    simulation.step()
    return make_bold(output)


def observe(simulation: InteractiveContext, **_: Any) -> str:
    """Observe the state of the game."""
    output = f"\nHere is the state of the world:\n\n{simulation.get_population()}\n\n"
    return make_bold(output)


def show_map(simulation: InteractiveContext, **_: Any) -> str:
    """Display the map."""
    cell_width = 3

    coordinates = simulation.get_population()[[X, Y, IS_VILLAGE]].reset_index()

    # Determine the maximum x and y coordinates to define the grid size
    max_x = coordinates[X].max() + 1
    max_y = coordinates[Y].max() + 1

    # Create a grid and fill it with the index values
    grid = np.full((max_y, max_x), "---", dtype=f"U{cell_width}")
    grid[coordinates[Y].values, coordinates[X].values] = (
        coordinates[IS_VILLAGE].map({True: " V ", False: "---"}).values
    )

    # Add border to grid
    vertical_border = np.full((max_y, 1), "|", dtype="U1")
    grid = np.hstack([vertical_border, grid, vertical_border])

    # Generate the string representation of the grid
    horizontal_border = "\n+" + "-" * (max_x * (cell_width + 1) + 1) + "+\n"
    map_string = (
        horizontal_border + "\n".join([" ".join(row) for row in grid]) + horizontal_border
    )

    map_string = map_string.replace(" V ", "[bold green] V [/bold green]")

    output = f"\n{make_bold('Here is a map of the world:')}\n{map_string}\n"
    return output


def test(simulation: InteractiveContext, **_: Any) -> str:
    """Run a test."""
    output = f"\nRunning a test.\n\n"
    a = np.array([[1, 2, 3], [4, 555, 6], [7, 8, 9], [0, -1, -2]])
    output += f"{a}\n\n"
    return make_bold(output)


def invalid_input(simulation: InteractiveContext, user_input: str, **_: Any) -> str:
    """Handle invalid input from the player."""
    output = f"\nInvalid command provided: '{user_input}'\n\n"
    return make_bold(output)


def finish_game(simulation: InteractiveContext, debug: bool = False, **_: Any) -> str:
    """Finish the game."""
    if debug:
        simulation.finalize()
    return ""
