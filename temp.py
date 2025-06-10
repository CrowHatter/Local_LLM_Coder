def quicksort(arr):
    """
    Sorts an array of integers using the quicksort algorithm.

    Args:
        arr (list): The array of integers to be sorted.

    Returns:
        list: The sorted array of integers.
    """
    if len(arr) <= 1:
        return arr

    pivot = arr[0]
    less = [x for x in arr[1:] if x <= pivot]
    greater = [x for x in arr[1:] if x > pivot]

    return quicksort(less) + [pivot] + quicksort(greater)

# Test the quicksort function
arr = [3, 2, 1, 4, 5, 6]
print(quicksort(arr))