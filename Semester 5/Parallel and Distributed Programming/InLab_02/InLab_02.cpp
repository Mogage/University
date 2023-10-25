#include <iostream>
#include <thread>
#include <vector>
#include <random>
#include <algorithm>
#include <iterator>
#include <ctime>
#include <chrono>

const int size = 100000000;

//int A[size];
//int B[size];
//int C[size];

std::vector<int> A(size);
std::vector<int> B(size);
std::vector<int> C(size);

void runLinear(int start, int end)
{
    for (int i = start; i < end; ++i)
    {
        C[i] = A[i] + B[i];
    }
}

void runCyclic(int id, int p) 
{
    for (int i = id; i < size; i += p) 
    {
        C[i] = A[i] + B[i];
    }
}

void printVector(std::vector<int> toPrint)
{
    for (const auto element : toPrint)
    {
        std::cout << element << " ";
    }
    std::cout << '\n';
}

int main()
{
    const int p = 4;
    int threadTasks = size / p;
    int reminder = size % p;
    int start = 0;
    int end;
    std::thread th[p];

    std::random_device rnd_device;
    std::mt19937 mersenne_engine{ rnd_device() }; 
    std::uniform_int_distribution<int> dist{ 1, 52 };

    auto gen = [&dist, &mersenne_engine]() {
        return dist(mersenne_engine);
        };

    std::generate(std::begin(A), std::end(A), gen);
    std::generate(std::begin(B), std::end(B), gen);

    auto t_start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < p; ++i)
    {
        end = start + threadTasks;
        if (reminder > 0)
        {
            --reminder;
            ++end;
        }
        th[i] = std::thread(runLinear, start, end);
        start = end;
    }

   /* int id, pas;
    for (int i = 0; i < p; ++i) {
        th[i] = std::thread(runCyclic, i, p);
    }*/

    for (int i = 0; i < p; ++i) 
    {
        th[i].join();
    }
    
    auto t_end = std::chrono::high_resolution_clock::now();
    double elapsed_time_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();

    std::cout << "elapsed_time_ms = " << elapsed_time_ms << "\n";

    /*printVector(A);
    printVector(B);
    printVector(C);*/
}
