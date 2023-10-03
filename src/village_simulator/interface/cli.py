import click
from vivarium import InteractiveContext

from village_simulator.interface import actions
from village_simulator import paths


@click.group()
def village_simulator():
    """ A command line utility for playing the Village Simulator."""
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

    flavor_message = "It is the year 3000 BC. You are the chieftain of a small village.\n"
    base_message = (
        "What would you like to do now?"
        "\n\nOptions:\n\t- observe (o)\n\t- step (s)\n\t- exit (x)\n"
    )

    playing = True
    while playing:
        user_input = input(flavor_message + base_message)

        match user_input:
            case "observe" | "o":
                action = actions.observe
            case "step" | "s":
                action = actions.step
            case "exit" | "x":
                playing = False
                action = actions.finish_game
            case _:
                action = actions.invalid_input

        flavor_message = action(simulation, user_input=user_input, debug=debug)


