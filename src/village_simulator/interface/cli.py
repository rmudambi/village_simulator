import click
from rich.console import Console
from vivarium import InteractiveContext

from village_simulator import paths
from village_simulator.interface import actions


@click.group()
def village_simulator():
    """A command line utility for playing the Village Simulator."""
    pass


@village_simulator.command()
@click.option(
    "--debug",
    "-d",
    is_flag=True,
    help="For use when debugging.",
)
def play(debug: bool):
    """Play a Village Simulator game."""
    simulation = InteractiveContext(paths.GAME_SPECIFICATION)
    console = Console()

    flavor_message = "\nIt is the year 3000 BC. You are the chieftain of a small village.\n"
    base_message = (
        "What would you like to do now?"
        "\n\nOptions:"
        "\n  - observe (o)"
        "\n  - show map (m)"
        "\n  - step (s)"
        "\n  - exit (x)"
        "\n\n>>> "
    )

    playing = True
    while playing:
        console.rule()
        user_input = console.input(f"[bold]{flavor_message}[/bold]{base_message}")

        match user_input:
            case "observe" | "o":
                action = actions.observe
            case "show map" | "m":
                action = actions.show_map
            case "step" | "s":
                action = actions.step
            case "exit" | "x":
                playing = False
                action = actions.finish_game
            case _:
                action = actions.invalid_input

        flavor_message = action(simulation, user_input=user_input, debug=debug)

    console.print("Thank you for playing! Goodbye!")
