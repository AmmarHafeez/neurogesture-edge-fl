"""Gesture label to command mapping."""


DEFAULT_COMMANDS = {
    0: "rest",
    1: "gesture_1",
    2: "gesture_2",
}


def command_for_label(label: int) -> str:
    """Map a predicted label to a local command name."""
    return DEFAULT_COMMANDS.get(label, "unknown")
