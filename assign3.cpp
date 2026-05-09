#include <bits/stdc++.h>
#include <omp.h>
using namespace std;

int min_seq(vector<int> &arr) {
    int mn = INT_MAX;
    for (int x : arr)
        if (x < mn) mn = x;
    return mn;
}

int max_seq(vector<int> &arr) {
    int mx = INT_MIN;
    for (int x : arr)
        if (x > mx) mx = x;
    return mx;
}

long long sum_seq(vector<int> &arr) {
    long long sum = 0;
    for (int x : arr)
        sum += x;
    return sum;
}

double avg_seq(vector<int> &arr) {
    return (double)sum_seq(arr) / arr.size();
}

int min_par(vector<int> &arr) {
    int mn = INT_MAX;
    #pragma omp parallel for reduction(min:mn)
    for (int i = 0; i < arr.size(); i++)
        if (arr[i] < mn) mn = arr[i];
    return mn;
}

int max_par(vector<int> &arr) {
    int mx = INT_MIN;
    #pragma omp parallel for reduction(max:mx)
    for (int i = 0; i < arr.size(); i++)
        if (arr[i] > mx) mx = arr[i];
    return mx;
}

long long sum_par(vector<int> &arr) {
    long long sum = 0;
    #pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < arr.size(); i++)
        sum += arr[i];
    return sum;
}

double avg_par(vector<int> &arr) {
    return (double)sum_par(arr) / arr.size();
}

int main() {
    int n;
    cin >> n;

    vector<int> arr(n);
    for (int i = 0; i < n; i++)
        arr[i] = rand() % 1000;

    double start, end;

    start = omp_get_wtime();
    int mn1 = min_seq(arr);
    int mx1 = max_seq(arr);
    long long sum1 = sum_seq(arr);
    double avg1 = avg_seq(arr);
    end = omp_get_wtime();

    double seq_time = end - start;

    start = omp_get_wtime();
    int mn2 = min_par(arr);
    int mx2 = max_par(arr);
    long long sum2 = sum_par(arr);
    double avg2 = avg_par(arr);
    end = omp_get_wtime();

    double par_time = end - start;

    cout << "Sequential Results:\n";
    cout << "Min = " << mn1 << "\n";
    cout << "Max = " << mx1 << "\n";
    cout << "Sum = " << sum1 << "\n";
    cout << "Avg = " << avg1 << "\n";
    cout << "Time = " << seq_time << " seconds\n";

    cout << "\nParallel Results:\n";
    cout << "Min = " << mn2 << "\n";
    cout << "Max = " << mx2 << "\n";
    cout << "Sum = " << sum2 << "\n";
    cout << "Avg = " << avg2 << "\n";
    cout << "Time = " << par_time << " seconds\n";

    return 0;
}