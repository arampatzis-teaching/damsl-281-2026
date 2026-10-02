"""Two versions of the same operation, and one function with hidden state."""

total = 0.0


def add_bias(values, b):
    """Add b to every entry, in place. The caller's list is changed."""
    for i in range(len(values)):
        values[i] += b
    return values


def add_bias_pure(values, b):
    """Return a new list with b added to every entry."""
    return [v + b for v in values]


def accumulate(x):
    """Add x to a variable outside the function and return the running sum."""
    global total
    total += x
    return total
