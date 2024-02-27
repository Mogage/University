#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <fstream>
#include <chrono>
#include <string>

using namespace std;

int n, m, kn, n_threads;

int* convolutionMatrix;
int* inputMatrix;
int* outputMatrix;

void readMatrix(string filename, int n, int m, int* matrix)
{
    ifstream fin(filename);
    int x, y, index;
    fin >> x >> y;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            fin >> x;
            index = i * m + j;
            matrix[index] = x;
        }
    }
    fin.close();
}

void writeMatrix(string filename, int n, int m, int* matrix) {

    ofstream fout(filename);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++)
            fout << matrix[i * m + j] << " ";
        fout << '\n';
    }
    fout.close();
}

void checkCompliance(string filename, int n, int m, int* matrix) {
    ifstream in(filename);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++)
        {
            int x;
            in >> x;
            if (x != matrix[i * m + j]) {
                cout << "Not equal";
                return;
            }
        }
    }
    in.close();
}

__global__ void convolute(int* inputMat, int* convMat, int* outputMat, int n, int m, int kn)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < m) {
        int res = 0;
        for (int i = 0; i < kn; i++)
        {
            for (int j = 0; j < kn; j++) {
                int x = min(max(row - kn / 2 + i, 0), n - 1);
                int y = min(max(col - kn / 2 + j, 0), m - 1);

                res += inputMat[x * m + y] * convMat[i * kn + j];
            }
        }
        outputMat[row * m + col] = res;
    }
}

__host__ void cudaConvolute() {
    int* cudaInputMat;
    int* cudaConv;
    int* cudaOutput;

    cudaMalloc((void**)&cudaInputMat, n * m * sizeof(int));
    cudaMalloc((void**)&cudaOutput, n * m * sizeof(int));
    cudaMalloc((void**)&cudaConv, kn * kn * sizeof(int));

    cudaMemcpy(cudaInputMat, inputMatrix, n * m * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(cudaConv, convolutionMatrix, kn * kn * sizeof(int), cudaMemcpyHostToDevice);

    if (n_threads == 0)
    {
        dim3 blockSize(1, 1);
        dim3 gridSize(n, m);
        convolute << <gridSize, blockSize >> > (cudaInputMat, cudaConv, cudaOutput, n, m, kn);
    }
    else
    {
        dim3 blockSize(n_threads, n_threads);
        dim3 gridSize((n + blockSize.x - 1) / blockSize.x, (m + blockSize.y - 1) / blockSize.y);
        convolute << <gridSize, blockSize >> > (cudaInputMat, cudaConv, cudaOutput, n, m, kn);
    }

    cudaMemcpy(outputMatrix, cudaOutput, n * m * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(cudaInputMat);
    cudaFree(cudaConv);
    cudaFree(cudaOutput);
}

int main(int argc, char* argv[])
{
    n_threads = stoi(argv[1]);
    n = stoi(argv[2]);
    m = stoi(argv[3]);
    kn = stoi(argv[4]);

    string INPUT_FILE = "inputs\\" + to_string(n) + "x" + to_string(n) + ".txt";
    string CONVOLUTION_FILE = "inputs\\convolution.txt";
    string OUTPUT_FILE = "outputs\\" + to_string(n) + ".txt";

    inputMatrix = new int[n * m];
    outputMatrix = new int[n * m];
    convolutionMatrix = new int[kn * kn];

    readMatrix(INPUT_FILE, n, m, inputMatrix);
    readMatrix(CONVOLUTION_FILE, kn, kn, convolutionMatrix);

    auto startTime = chrono::high_resolution_clock::now();

    cudaConvolute();

    auto endTime = chrono::high_resolution_clock::now();
    auto duration = chrono::duration<double, milli>(endTime - startTime).count();

    if (n_threads == 0)
    {
        writeMatrix(OUTPUT_FILE, n, m, outputMatrix);
    }
    else
    {
        checkCompliance(OUTPUT_FILE, n, m, outputMatrix);
    }
    cout << to_string(duration);

    delete[] outputMatrix;
    delete[] inputMatrix;
    delete[] convolutionMatrix;


    return 0;
}