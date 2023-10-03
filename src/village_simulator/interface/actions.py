from typing import Any

from vivarium import InteractiveContext


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
