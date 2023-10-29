from vivarium import InteractiveContext


def get_date_text(simulation: InteractiveContext) -> str:
    date = simulation.current_time.strftime("%b %d, %Y")
    return f"It now is {date}.\n"


def make_bold(text: str) -> str:
    return f"[bold]{text}[/bold]"
