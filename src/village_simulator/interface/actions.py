from typing import Any

import numpy as np
from vivarium import InteractiveContext

from village_simulator.simulation.map import X, Y


def _get_date_flavor_text(simulation: InteractiveContext) -> str:
    date = simulation.current_time.strftime("%b %d, %Y")
    return f"It now is {date}.\n"


def step(simulation: InteractiveContext, **_: Any) -> str:
    """Progress the game."""
    print(f"\nAdvancing the game by {simulation.configuration.time.step_size} days.\n")
    simulation.step()
    return _get_date_flavor_text(simulation)


def observe(simulation: InteractiveContext, **_: Any) -> str:
    """Observe the state of the game."""
    print(f"\nHere is the state of the world:\n\n{simulation.get_population()}\n")
    return _get_date_flavor_text(simulation)


def show_map(simulation: InteractiveContext, **_: Any) -> str:
    """Display the map."""
    coordinates = simulation.get_population()[[X, Y]].reset_index()

    # Determine the maximum x and y coordinates to define the grid size
    max_x = coordinates[X].max()
    max_y = coordinates[Y].max()

    # Create a grid and fill it with the index values
    grid = np.full((max_y + 1, max_x + 1), "-", dtype=str)
    grid[coordinates[Y].values, coordinates[X].values] = "V"

    # Generate the string representation of the grid
    map_string = "\n".join([" ".join(row) for row in grid])

    print(f"\nHere is a map of the world:\n\n{map_string}\n")
    return _get_date_flavor_text(simulation)


def invalid_input(simulation: InteractiveContext, user_input: str, **_: Any) -> str:
    """Handle invalid input from the player."""
    print(f"\nInvalid command provided: '{user_input}'\n")
    return _get_date_flavor_text(simulation)


def finish_game(simulation: InteractiveContext, debug: bool = False, **_: Any) -> str:
    """Finish the game."""
    if debug:
        simulation.finalize()
    print("Thank you for playing! Good bye!")
    return ""
