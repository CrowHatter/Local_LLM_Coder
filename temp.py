def selection_sort(arr):
    """
    Sorts the input array using the selection sort algorithm.

    Args:
        arr (list): The input array to be sorted.

    Returns:
        list: The sorted array.
    """
    for i in range(len(arr) - 1):
        """
        Find the minimum element in the unsorted portion of the array.
        """
        min_pos = i
        for j in range(i + 1, len(arr)):
            """
            Compare the current element with the minimum element.
            """
            if arr[j] < arr[min_pos]:
                """
                Update the minimum element index if the current element is smaller.
                """
                min_pos = j
        """
        Swap the minimum element with the first element of the unsorted portion.
        """
        arr[i], arr[min_pos] = arr[min_pos], arr[i]
    return arr
