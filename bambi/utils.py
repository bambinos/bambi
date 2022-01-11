from numpy.linalg import matrix_rank


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


def check_full_rank(matrix):
    """Checks if a matrix is full rank

    Parameters
    ----------
    matrix: numpy.array
        A 2-dimensional NumPy array that represents a design matrix

    Returns
    -------
    None
    """
    if matrix_rank(matrix) < matrix.shape[1]:
        raise ValueError(
            "Design matrix for common effects is not full-rank. "
            "Bambi does not support sparse settings when automatic priors are obtained via MLE."
        )
