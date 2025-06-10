void selection_sort(int* arr, int size){
    for(int i=0; i<size-1; i++){
        int min_pos = i;
        for(int j=i+1; j<size; j++){
            if(arr[j] < arr[min_pos])
                min_pos = j;
        }
        //swap
        int tmp = arr[min_pos];
        arr[min_pos] = arr[i];
        arr[i] = tmp;
    }
}

void bubble_sort(int* arr, int size){
    size--;
    while(size){
        for(int i=0, j=i+1; i<size, j<=size; i++, j++){
            //swap
            if(arr[i] > arr[j]){
                int tmp = arr[i];
                arr[i] = arr[j];
                arr[j] = tmp;
            }
        }
        size--;
    }
}


int Partition(int* arr, int front, int end){
    int pivot = arr[end];
    //left part end
    int left_end = front-1;
    //right part end
    int right_end = front;
    for(; right_end<end; right_end++){
        //check if the right_end is bigger, if not change the value with the left_end
        if(arr[right_end] < pivot){
            //vacate a space for left_end
            left_end++;
            //swap value
            int tmp = arr[right_end];
            arr[right_end] = arr[left_end];
            arr[left_end] = tmp;
        }
    }
    //After the partition place the pivot in the middle
    //swap the first element of right part with pivot
    arr[end] = arr[left_end+1];
    arr[left_end+1] = pivot;
    
    //return the position of pivot
    return left_end+1;
}
void quick_sort(int* arr, int front, int end){
    //boundary
    if(!arr)
        return;
    
    //main
    if(front < end){
        int pivot = Partition(arr, front, end);
        quick_sort(arr, front, pivot-1);
        quick_sort(arr, pivot+1, end);
    }
}