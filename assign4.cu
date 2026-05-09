#include <iostream>
#include <cuda_runtime.h>
#include <ctime>

using namespace std;

__global__ void vectorAddCUDA(int *a, int *b, int *c, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        c[idx] = a[idx] + b[idx];
}

void vectorAddCPU(int *a, int *b, int *c, int n)
{
    for (int i = 0; i < n; i++)
        c[i] = a[i] + b[i];
}

__global__ void matrixMulCUDA(int *a, int *b, int *c, int N)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N)
    {
        int sum = 0;
        for (int k = 0; k < N; k++)
            sum += a[row * N + k] * b[k * N + col];
        c[row * N + col] = sum;
    }
}

void matrixMulCPU(int *a, int *b, int *c, int N)
{
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
        {
            int sum = 0;
            for (int k = 0; k < N; k++)
                sum += a[i * N + k] * b[k * N + j];
            c[i * N + j] = sum;
        }
}

int main()
{
    int vecSize = 1 << 20;
    int matrixSize = 512;

    // cout << "Enter vector size: ";
    // cin >> vecSize;
    // cout << "Enter matrix size (N for NxN): ";
    // cin >> matrixSize;

    int *h_a = new int[vecSize];
    int *h_b = new int[vecSize];
    int *h_c_cpu = new int[vecSize];
    int *h_c_gpu = new int[vecSize];

    for (int i = 0; i < vecSize; i++)
    {
        h_a[i] = rand() % 100;
        h_b[i] = rand() % 100;
    }

    // cout << "Enter vector A:\n";
    // for (int i = 0; i < vecSize; i++)
    //     cin >> h_a[i];
    // cout << "Enter vector B:\n";
    // for (int i = 0; i < vecSize; i++)
    //     cin >> h_b[i];

    clock_t start = clock();
    vectorAddCPU(h_a, h_b, h_c_cpu, vecSize);
    clock_t end = clock();

    cout << "[Vector CPU Time] "
         << (double)(end - start) * 1000 / CLOCKS_PER_SEC << " ms\n";

    int *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, vecSize * sizeof(int));
    cudaMalloc(&d_b, vecSize * sizeof(int));
    cudaMalloc(&d_c, vecSize * sizeof(int));

    cudaMemcpy(d_a, h_a, vecSize * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, vecSize * sizeof(int), cudaMemcpyHostToDevice);

    cudaEvent_t startEvent, stopEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);

    cudaEventRecord(startEvent);

    vectorAddCUDA<<<(vecSize + 255) / 256, 256>>>(d_a, d_b, d_c, vecSize);

    cudaEventRecord(stopEvent);
    cudaEventSynchronize(stopEvent);

    float gpuTime = 0;
    cudaEventElapsedTime(&gpuTime, startEvent, stopEvent);

    cudaMemcpy(h_c_gpu, d_c, vecSize * sizeof(int), cudaMemcpyDeviceToHost);

    cout << "[Vector GPU Time] " << gpuTime << " ms\n\n";

    // cout << "Vector Result:\n";
    // for (int i = 0; i < vecSize; i++)
    //     cout << h_c_gpu[i] << " ";
    // cout << "\n\n";

    int N = matrixSize;
    int size = N * N;

    int *matA = new int[size];
    int *matB = new int[size];
    int *matC_cpu = new int[size];
    int *matC_gpu = new int[size];

    for (int i = 0; i < size; i++)
    {
        matA[i] = rand() % 10;
        matB[i] = rand() % 10;
    }

    // cout << "Enter Matrix A:\n";
    // for (int i = 0; i < size; i++)
    //     cin >> matA[i];
    // cout << "Enter Matrix B:\n";
    // for (int i = 0; i < size; i++)
    //     cin >> matB[i];

    start = clock();
    matrixMulCPU(matA, matB, matC_cpu, N);
    end = clock();

    cout << "[Matrix CPU Time] "
         << (double)(end - start) * 1000 / CLOCKS_PER_SEC << " ms\n";

    int *d_A, *d_B, *d_C;
    size_t bytes = size * sizeof(int);

    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    cudaMemcpy(d_A, matA, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, matB, bytes, cudaMemcpyHostToDevice);

    dim3 threads(16, 16);
    dim3 blocks((N + 15) / 16, (N + 15) / 16);

    cudaEventRecord(startEvent);

    matrixMulCUDA<<<blocks, threads>>>(d_A, d_B, d_C, N);

    cudaEventRecord(stopEvent);
    cudaEventSynchronize(stopEvent);

    cudaEventElapsedTime(&gpuTime, startEvent, stopEvent);

    cudaMemcpy(matC_gpu, d_C, bytes, cudaMemcpyDeviceToHost);

    cout << "[Matrix GPU Time] " << gpuTime << " ms\n";

    // cout << "Matrix Result:\n";
    // for (int i = 0; i < N; i++)
    // {
    //     for (int j = 0; j < N; j++)
    //         cout << matC_gpu[i * N + j] << " ";
    //     cout << "\n";
    // }

    delete[] h_a;
    delete[] h_b;
    delete[] h_c_cpu;
    delete[] h_c_gpu;
    delete[] matA;
    delete[] matB;
    delete[] matC_cpu;
    delete[] matC_gpu;

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}