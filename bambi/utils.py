def listify(obj):
    """Wrap all non-list or tuple objects in a list.

    Provides a simple way to accept flexible arguments.
    """
    if obj is None:
        return []
    else:
        return obj if isinstance(obj, (list, tuple, type(None))) else [obj]


def spacify(string):
    """Add 2 spaces to the beginning of each line in a multi-line string."""
    return "  " + "  ".join(string.splitlines(True))


def multilinify(sequence, sep=","):
    """Make a multi-line string out of a sequence of strings."""
    sep += "\n"
    return "\n" + sep.join(sequence)
