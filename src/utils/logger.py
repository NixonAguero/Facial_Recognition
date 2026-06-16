from datetime import datetime


def log(message: str) -> None:
    """Print a simple log message with the current time."""
    current_time = datetime.now().strftime("%H:%M:%S")
    print(f"[{current_time}] {message}", flush=True)
