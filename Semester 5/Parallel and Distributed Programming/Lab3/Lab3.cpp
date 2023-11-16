#include <iostream>
#include <fstream>
#include <string>
#include <random>
#include <chrono>
#include <vector>
#include "mpi.h"

std::string fileName = "matrix.txt";
std::chrono::steady_clock::time_point startTime1;
std::chrono::steady_clock::time_point startTime2;
std::chrono::duration<double, std::milli> duration1;
std::chrono::duration<double, std::milli> duration2;
std::ifstream fin(fileName);

const int rowSize = 1000;
const int columnSize = 1000;
const int smallSize = 3;
int halfSmallSize = smallSize / 2;

int flatBigMatrix[rowSize * columnSize];
int bigMatrix[rowSize][columnSize];
int smallMatrix[smallSize][smallSize];

void createFile();
void readFromFile(bool);
void writeMatrixToFile(int matrix[rowSize][columnSize]);

void copyVector(int[], int[]);
void printVector(int[]);

//int main(int argc, char* argv[])
//{
//    std::srand(std::time(0));
//
//    int myid, numprocs, namelen;
//    char processor_name[MPI_MAX_PROCESSOR_NAME];
//    int temp;
//
//    MPI_Init(&argc, &argv);
//    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
//    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
//    MPI_Get_processor_name(processor_name, &namelen);
//
//    if (std::strcmp(argv[1], "1") == 0)
//    {
//        if (myid == 0)
//        {
//            createFile();
//        }
//        MPI_Finalize();
//        return 0;
//    }
//    else if (myid == 0)
//    {
//        startTime1 = std::chrono::high_resolution_clock::now();
//        readFromFile(true);
//
//        MPI_Bcast(&smallMatrix, smallSize * smallSize, MPI_INT, 0, MPI_COMM_WORLD);
//        
//    }
//
//    int batchSize = rowSize / numprocs;
//    int* auxBatchMatrix = new int[batchSize * columnSize];
//    int* auxBigMatrix = new int[batchSize * columnSize];
//    startTime2 = std::chrono::high_resolution_clock::now();
//    MPI_Scatter(flatBigMatrix, batchSize * columnSize, MPI_INT, auxBatchMatrix, batchSize * columnSize, MPI_INT, 0, MPI_COMM_WORLD);
//    MPI_Bcast(&smallMatrix, smallSize * smallSize, MPI_INT, 0, MPI_COMM_WORLD);
//    int buffer[3][columnSize];
//
//    if (myid == 0)
//    {
//        MPI_Send(&auxBatchMatrix[(batchSize - 1) * columnSize], columnSize, MPI_INT, myid + 1, 0, MPI_COMM_WORLD);
//        MPI_Recv(&buffer[2], columnSize, MPI_INT, myid + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
//        copyVector(buffer[0], auxBatchMatrix);
//    }
//    else if (myid < numprocs - 1)
//    {
//        MPI_Recv(buffer[0], columnSize, MPI_INT, myid - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
//        MPI_Send(auxBatchMatrix, columnSize, MPI_INT, myid - 1, 0, MPI_COMM_WORLD);
//
//        MPI_Send(&auxBigMatrix[(batchSize - 1) * columnSize], columnSize, MPI_INT, myid + 1, 0, MPI_COMM_WORLD);
//        MPI_Recv(&buffer[2], columnSize, MPI_INT, myid + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
//    }
//    else
//    {
//        MPI_Recv(buffer[0], columnSize, MPI_INT, myid - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
//        MPI_Send(auxBatchMatrix, columnSize, MPI_INT, myid - 1, 0, MPI_COMM_WORLD);
//        copyVector(buffer[2], &auxBatchMatrix[(batchSize - 1) * columnSize]);
//    }
//
//    for (int i = 0; i < batchSize; i++)
//    {
//        copyVector(buffer[1], auxBatchMatrix + (i * columnSize));
//        for (int j = 0; j < columnSize; j++)
//        {
//            auxBigMatrix[i * columnSize + j] = 0;
//            for (int k = 0; k < 2; k++)
//            {
//                for (int l = 0; l < smallSize; l++)
//                {
//                    int columnIndex = std::min(std::max(j + l - halfSmallSize, 0), columnSize - 1);
//                    auxBigMatrix[i * columnSize + j] += buffer[k][columnIndex] * smallMatrix[k][l];
//                }
//            }
//            for (int l = 0; l < smallSize; l++)
//            {
//                int columnIndex = std::min(std::max(j + l - halfSmallSize, 0), columnSize - 1);
//                if (i >= batchSize - 1) {
//                    auxBigMatrix[i * columnSize + j] += buffer[2][columnIndex] * smallMatrix[2][l];
//                    continue;
//                }
//                auxBigMatrix[i * columnSize + j] += auxBatchMatrix[(i + 1) * columnSize + columnIndex] * smallMatrix[2][l];
//            }
//        }
//        copyVector(buffer[0], buffer[1]);
//    }
//
//    MPI_Gather(auxBigMatrix, batchSize * columnSize, MPI_INT, flatBigMatrix, batchSize * columnSize, MPI_INT, 0, MPI_COMM_WORLD);
//
//    if (myid == 0)
//    {
//        auto endTime2 = std::chrono::high_resolution_clock::now();
//        duration2 = endTime2 - startTime2;
//        std::cout << duration2.count() << '\n';
//        std::ofstream fout("result.txt");
//        for (int i = 0; i < rowSize; i++)
//        {
//            for (int j = 0; j < columnSize; j++)
//            {
//                fout << flatBigMatrix[i * columnSize + j] << " ";
//            }
//            fout << '\n';
//        }
//        auto endTime1 = std::chrono::high_resolution_clock::now();
//        duration1 = endTime1 - startTime1;
//        std::cout << duration1.count() << '\n';
//    }
//
//    MPI_Finalize();
//    return 0;
//}


int main(int argc, char* argv[])
{
    std::srand(std::time(0));

    int myid, numprocs, namelen;
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int temp;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    MPI_Get_processor_name(processor_name, &namelen);

    if (std::strcmp(argv[1], "1") == 0)
    {
        if (myid == 0)
        {
            createFile();
        }
    }
    else if (myid == 0)
    {
        int batchSize = rowSize / (numprocs - 1);
        int batchReminder = rowSize % (numprocs - 1);
        int start = 0;
        int end;

        startTime1 = std::chrono::high_resolution_clock::now();
        readFromFile(false);

        MPI_Bcast(&smallMatrix, smallSize * smallSize, MPI_INT, 0, MPI_COMM_WORLD);

        startTime2 = std::chrono::high_resolution_clock::now();
        for (int procIndex = 1; procIndex < numprocs; procIndex++)
        {
            end = start + batchSize;
            if (batchReminder > 0)
            {
                end++;
                batchReminder--;
            }
            for (int i = start; i < end; i++)
            {
                for (int j = 0; j < columnSize; j++)
                {
                    fin >> temp;
                    bigMatrix[i][j] = temp;
                }
            }
            MPI_Send(&start, 1, MPI_INT, procIndex, 0, MPI_COMM_WORLD);
            MPI_Send(&end, 1, MPI_INT, procIndex, 0, MPI_COMM_WORLD);
            MPI_Send(bigMatrix[start], (end - start) * columnSize, MPI_INT, procIndex, 0, MPI_COMM_WORLD);
            start = end;
        }

        for (int procIndex = 1; procIndex < numprocs; procIndex++)
        {
            MPI_Recv(&start, 1, MPI_INT, procIndex, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(&end, 1, MPI_INT, procIndex, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(bigMatrix[start], (end - start) * columnSize, MPI_INT, procIndex, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        auto endTime = std::chrono::high_resolution_clock::now();
        duration2 = endTime - startTime2;
        std::cout << duration2.count() << '\n';

        writeMatrixToFile(bigMatrix);

        endTime = std::chrono::high_resolution_clock::now();
        duration1 = endTime - startTime1;
        std::cout << duration1.count() << '\n';
    }
    else
    {
        MPI_Bcast(&smallMatrix, smallSize * smallSize, MPI_INT, 0, MPI_COMM_WORLD);

        int start, end;
        MPI_Recv(&start, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&end, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        int* auxBigMatrix = new int[(end - start) * columnSize];
        MPI_Recv(auxBigMatrix, (end - start) * columnSize, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        int sum;
        int rowIndex;
        int columnIndex;

        int buffer[3][columnSize];

        if (myid == 1)
        {
            MPI_Send(&auxBigMatrix[(end - start - 1) * columnSize], columnSize, MPI_INT, myid + 1, 0, MPI_COMM_WORLD);
            MPI_Recv(&buffer[2], columnSize, MPI_INT, myid + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        else if (myid < numprocs - 1)
        {
            MPI_Recv(buffer[0], columnSize, MPI_INT, myid - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Send(auxBigMatrix, columnSize, MPI_INT, myid - 1, 0, MPI_COMM_WORLD);

            MPI_Send(&auxBigMatrix[(end - start - 1) * columnSize], columnSize, MPI_INT, myid + 1, 0, MPI_COMM_WORLD);
            MPI_Recv(&buffer[2], columnSize, MPI_INT, myid + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        else
        {
            MPI_Recv(buffer[0], columnSize, MPI_INT, myid - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Send(auxBigMatrix, columnSize, MPI_INT, myid - 1, 0, MPI_COMM_WORLD);
        }

        if (myid == 1)
        {
            copyVector(buffer[0], auxBigMatrix);
        }
        else if (myid == numprocs - 1)
        {
            copyVector(buffer[2], &auxBigMatrix[(end - start - 1) * columnSize]);
        }

        for (int i = start; i < end; i++)
        {
            copyVector(buffer[1], auxBigMatrix + ((i - start) * columnSize));
            for (int j = 0; j < columnSize; j++)
            {
                sum = 0;
                for (int k = 0; k < 2; k++)
                {
                    for (int l = 0; l < smallSize; l++)
                    {
                        columnIndex = std::min(std::max(j + l - halfSmallSize, 0), columnSize - 1);
                        sum += buffer[k][columnIndex] * smallMatrix[k][l];
                    }
                }
                for (int l = 0; l < smallSize; l++)
                {
                    columnIndex = std::min(std::max(j + l - halfSmallSize, 0), columnSize - 1);
                    if (i >= end - 1) {
                        sum += buffer[2][columnIndex] * smallMatrix[2][l];
                        continue;
                    }
                    sum += auxBigMatrix[(i - start + 1) * columnSize + columnIndex] * smallMatrix[2][l];
                }
                auxBigMatrix[(i - start) * columnSize + j] = sum;
            }
            copyVector(buffer[0], buffer[1]);
        }

        MPI_Send(&start, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
        MPI_Send(&end, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
        MPI_Send(auxBigMatrix, (end - start) * columnSize, MPI_INT, 0, 0, MPI_COMM_WORLD);

        delete[] auxBigMatrix;
    }

    MPI_Finalize();
    return 0;
}



void createFile()
{
    std::ofstream fout(fileName);
    for (int i = 0; i < smallSize; i++)
    {
        for (int j = 0; j < smallSize; j++)
        {
            fout << rand() % 2 << " ";
        }
        fout << '\n';
    }
    for (int i = 0; i < rowSize; i++) {
        for (int j = 0; j < columnSize; j++) {
            fout << rand() % 100 << " ";
        }
        fout << '\n';
    }
    fout.close();
}

void readFromFile(bool isScatter)
{
    int temp;
    for (int i = 0; i < smallSize; i++) {
        for (int j = 0; j < smallSize; j++) {
            fin >> temp;
            smallMatrix[i][j] = temp;
        }
    }
    if (!isScatter)
    {
        return;
    }
    for (int i = 0; i < rowSize * columnSize; i++) {
        fin >> temp;
        flatBigMatrix[i] = temp;
    }
}

void writeMatrixToFile(int matrix[rowSize][columnSize])
{
    std::ofstream fout("result.txt");
    for (int i = 0; i < rowSize; i++) {
        for (int j = 0; j < columnSize; j++) {
            fout << matrix[i][j] << " ";
        }
        fout << '\n';
    }
    fout.close();
}

void copyVector(int newVector[], int oldVector[])
{
    for (int i = 0; i < columnSize; i++)
    {
        newVector[i] = oldVector[i];
    }
}

void printVector(int vectorToPrint[])
{
    for (int i = 0; i < columnSize; i++)
    {
        std::cout << vectorToPrint[i] << " ";
    }
    std::cout << '\n';
    std::cout << '\n';
}