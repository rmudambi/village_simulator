from typing import Any

import numpy as np
from vivarium import InteractiveContext

from village_simulator.constants import Columns
from village_simulator.interface.utilities import make_bold


def step(simulation: InteractiveContext, **_: Any) -> str:
    """Progress the game."""
    output = f"\nAdvancing the game by {simulation.configuration.time.step_size} days.\n\n"
    simulation.step()
    return make_bold(output)


def observe(simulation: InteractiveContext, **_: Any) -> str:
    """Observe the state of the game."""
    output = f"\nHere is the state of the world:\n\n{simulation.get_population()}\n\n"
    return make_bold(output)


def show_map(simulation: InteractiveContext, user_input: str, **kwargs: Any) -> str:
    """Display the map."""

    if user_input == "mv":
        mapped_column = Columns.IS_VILLAGE
        cell_width = 3
        formatter = {
            True: "[bold green] V [/bold green]",
            False: "---",
        }

    elif user_input == "mt":
        mapped_column = Columns.TERRAIN
        cell_width = 3
        formatter = {
            "mountain": "[bold blue] M [/bold blue]",
            "forest": "[bold green] F [/bold green]",
            "grassland": "[bold yellow] G [/bold yellow]",
            "desert": "[bold red] D [/bold red]",
        }

    else:
        return make_bold("\nAvailable map modes are 'mv' (village) and 'mt' (terrain).\n\n")

    coordinates = simulation.get_population()[
        [Columns.X, Columns.Y, mapped_column]
    ].reset_index()

    # Determine the maximum x and y coordinates to define the grid size
    max_x = coordinates[Columns.X].max() + 1
    max_y = coordinates[Columns.Y].max() + 1

    # Create a grid and fill it with the index values
    grid = np.full((max_y, max_x), "-", dtype="object")
    grid[coordinates[Columns.Y].values, coordinates[Columns.X].values] = (
        coordinates[mapped_column].map(formatter).values
    )

    # Add border to grid
    vertical_border = np.full((max_y, 1), "|", dtype="U1")
    grid = np.hstack([vertical_border, grid, vertical_border])

    # Generate the string representation of the grid
    horizontal_border = "\n+" + "-" * (max_x * (cell_width + 1) + 1) + "+\n"
    map_string = (
        horizontal_border + "\n".join([" ".join(row) for row in grid]) + horizontal_border
    )

    output = f"\n{make_bold('Here is a map of the world:')}\n{map_string}\n"

    return output


def invalid_input(_: InteractiveContext, user_input: str, **__: Any) -> str:
    """Handle invalid input from the player."""
    output = f"\nInvalid command provided: '{user_input}'\n\n"
    return make_bold(output)


def finish_game(simulation: InteractiveContext, debug: bool = False, **_: Any) -> str:
    """Finish the game."""
    if debug:
        simulation.finalize()
    return ""
