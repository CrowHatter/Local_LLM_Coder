def factorial(n):
    """
    Calculates the factorial of a given number.

    Args:
        n (int): The number for which the factorial is to be calculated.

    Returns:
        int: The factorial of the given number.

    Raises:
        ValueError: If the input is not an integer.

    Examples:
        >>> factorial(0)
        1
        >>> factorial(5)
        120
        >>> factorial(-1)
        ValueError: Negative numbers are not allowed.
    """
    if not isinstance(n, int):
        raise ValueError("Input must be an integer.")
    if n < 0:
        raise ValueError("Negative numbers are not allowed.")
    if n == 0:
        return 1
    else:
        return n * factorial(n-1)

print(factorial(10))
