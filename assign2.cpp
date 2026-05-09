#include <iostream>
#include <vector>
#include <algorithm>
#include <omp.h>
#include <cstdlib>

using namespace std;

void generate_array(vector<int>&arr , int size){
    arr.resize(size);
    for(int i=0;i<size;i++){
        arr[i] = rand() % 10000;
    }
}

void bubble_sort_sequential(vector<int>&arr){
    int n = arr.size();
    for(int i=0;i<n-1;i++){
        for(int j=0;j<n-i-1;j++){
            if(arr[j] > arr[j+1]){
                swap(arr[j] , arr[j+1]);
            }
        }
    }
}

void bubble_sort_parallel(vector<int>&arr){
    int n = arr.size();
    for(int i=0;i<n;i++){
        #pragma omp parallel for
        for(int j = (i % 2); j < n-1 ; j += 2){
            if(arr[j] > arr[j+1]){
                swap(arr[j] , arr[j+1]);
            }
        }
    }
}

void merge(vector<int>&arr , int left , int mid , int right){
    vector<int>temp(right - left + 1);
    int i = left, j = mid+1, k = 0;

    while(i <= mid && j <= right){
        if(arr[i] < arr[j]) temp[k++] = arr[i++];
        else temp[k++] = arr[j++];
    }

    while(i <= mid) temp[k++] = arr[i++];
    while(j <= right) temp[k++] = arr[j++];

    for(int i = 0; i < k; i++){
        arr[left + i] = temp[i];
    }
}

void mergesort_sequential(vector<int>&arr, int left , int right){
    if(left < right ){
        int mid = left + (right - left) / 2;
        mergesort_sequential(arr , left , mid);
        mergesort_sequential(arr , mid + 1, right);
        merge(arr , left , mid, right);
    }
}

void mergesort_parallel(vector<int>&arr , int left , int right, int depth){
    if(left < right){
        int mid = left + (right - left) / 2;
        
        if (depth < 4) { 
            #pragma omp task shared(arr)
            mergesort_parallel(arr , left , mid, depth + 1);
            
            #pragma omp task shared(arr)
            mergesort_parallel(arr , mid + 1, right, depth + 1);
            
            #pragma omp taskwait
        } else {
            mergesort_sequential(arr, left, mid);
            mergesort_sequential(arr, mid + 1, right);
        }
        merge(arr , left , mid , right);
    }
}

void start_mergesort_parallel(vector<int>&arr, int left, int right) {
    #pragma omp parallel
    {
        #pragma omp single
        {
            mergesort_parallel(arr, left, right, 0);
        }
    }
}

int main(){
    const int SIZE = 50000; 
    vector<int> original;
    generate_array(original, SIZE);

    vector<int> a1 = original, a2 = original;
    vector<int> a3 = original, a4 = original;

    double t1, t2;
    
    t1 = omp_get_wtime();
    bubble_sort_sequential(a1);
    t2 = omp_get_wtime();
    cout << "Sequential Bubble Sort Time: " << t2 - t1 << "s\n";
    
    t1 = omp_get_wtime();
    bubble_sort_parallel(a2);
    t2 = omp_get_wtime();
    cout << "Parallel Bubble Sort Time:   " << t2 - t1 << "s\n";
    
    t1 = omp_get_wtime();
    mergesort_sequential(a3, 0, SIZE-1);
    t2 = omp_get_wtime();
    cout << "Sequential Merge Sort Time:  " << t2 - t1 << "s\n";
   
    t1 = omp_get_wtime();
    start_mergesort_parallel(a4, 0, SIZE-1);
    t2 = omp_get_wtime();
    cout << "Parallel Merge Sort Time:    " << t2 - t1 << "s\n";

    return 0;
}