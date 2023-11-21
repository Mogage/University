#include <iostream>
#include <fstream>
#include <thread>
#include <barrier>
#include <chrono>
#include <string>
#include <vector>
#include <random>
#include <math.h>

const std::string fileName = "data.txt";
std::chrono::duration<double, std::milli> duration;

const int rowSize = 1000;
const int columnSize = 1000;
const int smallSize = 3;

std::vector<std::vector<int>> bigMatrix(rowSize, std::vector<int>(columnSize));
std::vector<std::vector<int>> smallMatrix(smallSize, std::vector<int>(smallSize));

void createFile();
void readFromFile();
void writeMatrixToFile(std::string);

void sequential();

void runOnRows(std::barrier<>&, int, int);
void multiThread(int);

void runProgram(char*[]);

void compareResults();

int main(int argc, char* argv[])
{
    std::srand(std::time(0));

    if (std::string(argv[1]) == "1")
    {
        createFile();
        return 0;
    }
    runProgram(argv);
    std::cout << duration.count();
    return 0;
}

void createFile()
{
    std::ofstream fout(fileName);
    for (int i = 0; i < rowSize; i++) {
        for (int j = 0; j < columnSize; j++) {
            fout << rand() % 101 << " ";
        }
        fout << '\n';
    }

    for (int i = 0; i < smallSize; i++) {
        for (int j = 0; j < smallSize; j++) {
            fout << rand() % 2 << " ";
        }
        fout << '\n';
    }
    fout.close();
}

void readFromFile()
{
    std::ifstream fin(fileName);
    int temp;
    for (int i = 0; i < rowSize; i++) {
        for (int j = 0; j < columnSize; j++) {
            fin >> temp;
            bigMatrix[i][j] = temp;
        }
    }

    for (int i = 0; i < smallSize; i++) {
        for (int j = 0; j < smallSize; j++) {
            fin >> temp;
            smallMatrix[i][j] = temp;
        }
    }
    fin.close();
}

void writeMatrixToFile(std::string fileName)
{
    std::ofstream fout(fileName);
    for (int i = 0; i < rowSize; i++) {
        for (int j = 0; j < columnSize; j++) {
            fout << bigMatrix[i][j] << " ";
        }
        fout << '\n';
    }
}

void sequential()
{
    int rowIndex;
    int columnIndex;
    int sum;
    int halfSmallSize = smallSize / 2;
    int noOfRowsBuffer = halfSmallSize + 1;
    std::vector<std::vector<int>> buffer = std::vector<std::vector<int>>(noOfRowsBuffer, std::vector<int>(columnSize));

    for (int i = 0; i < noOfRowsBuffer; i++)
    {
        buffer[i] = bigMatrix[0];
    }

    for (int i = 0; i < rowSize - 1; i++)
    {
        for (int j = 0; j < columnSize; j++)
        {
            sum = 0;
            for (int index1 = 0; index1 < noOfRowsBuffer; index1++)
            {
                for (int index2 = 0; index2 < smallSize; index2++)
                {
                    columnIndex = std::min(std::max(j - halfSmallSize + index2, 0), columnSize - 1);
                    sum += buffer[index1][columnIndex] * smallMatrix[index1][index2];
                }
            }
            for (int index1 = noOfRowsBuffer; index1 < smallSize; index1++)
            {
                rowIndex = std::min(std::max(i - halfSmallSize + index1, 0), rowSize - 1);
                for (int index2 = 0; index2 < smallSize; index2++)
                {
                    columnIndex = std::min(std::max(j - halfSmallSize + index2, 0), columnSize - 1);
                    sum += bigMatrix[rowIndex][columnIndex] * smallMatrix[index1][index2];
                }
            }
            bigMatrix[i][j] = sum;
        }

        for (int index1 = 0; index1 < noOfRowsBuffer - 1; index1++)
        {
            buffer[index1] = buffer[index1 + 1];
        }
        buffer[noOfRowsBuffer - 1] = bigMatrix[i + 1];
    }
    for (int j = 0; j < columnSize; j++)
    {
        sum = 0;
        for (int index1 = 0; index1 < smallSize; index1++)
        {
            rowIndex = std::min(index1, halfSmallSize);
            for (int index2 = 0; index2 < smallSize; index2++)
            {
                columnIndex = std::min(std::max(j - halfSmallSize + index2, 0), columnSize - 1);
                sum += buffer[rowIndex][columnIndex] * smallMatrix[index1][index2];
            }
        }
        bigMatrix[rowSize - 1][j] = sum;
    }
}

void runOnRows(std::barrier<>& barrier, int start, int end)
{
    int halfSmallSize = smallSize / 2;
    int noOfRowsBuffer = 3;
    int rowIndex;
    int columnIndex;
    int sum;
    std::vector<std::vector<int>> buffer = std::vector<std::vector<int>>(noOfRowsBuffer, std::vector<int>(columnSize));
    if (start == 0)
    {
        buffer[0] = bigMatrix[0];
    }
    else
    {
        buffer[0] = bigMatrix[start - 1];
    }
    buffer[1] = bigMatrix[start];
    if (end == rowSize)
    {
        buffer[2] = bigMatrix[rowSize - 1];
    }
    else
    {
        buffer[2] = bigMatrix[end];
    }
    barrier.arrive_and_wait();
    for (int i = start; i < end; i++)
    {
        for (int j = 0; j < columnSize; j++)
        {
            sum = 0;
            for (int index1 = 0; index1 < 2; index1++) {
                for (int index2 = 0; index2 < smallSize; index2++) {
                    columnIndex = std::min(std::max(j - halfSmallSize + index2, 0), columnSize - 1);
                    sum += buffer[index1][columnIndex] * smallMatrix[index1][index2];
                }
            }
            for (int index2 = 0; index2 < smallSize; index2++) {
                columnIndex = std::min(std::max(j - halfSmallSize + index2, 0), columnSize - 1);
                if (i >= end - 1) {
                    sum += buffer[2][columnIndex] * smallMatrix[2][index2];
                    continue;
                }
                sum += bigMatrix[i + 1][columnIndex] * smallMatrix[2][index2];
            }
            bigMatrix[i][j] = sum;
        }
        buffer[0] = buffer[1];
        buffer[1] = bigMatrix[std::min(i + 1, end - 1)];
    }
}

void multiThread(int threadCount)
{
    int batchSize = rowSize / threadCount;
    int batchReminder = rowSize % threadCount;
    int start = 0;
    int end;
    std::vector<std::thread> threads(threadCount);
    std::barrier barrier(threadCount);

    for (int i = 0; i < threadCount; i++)
    {
		end = start + batchSize;
        if (batchReminder > 0)
        {
			end++;
			batchReminder--;
		}
		threads[i] = std::thread(runOnRows, std::ref(barrier), start, end);
		start = end;
	}

    for (int i = 0; i < threadCount; i++)
    {
		threads[i].join();
	}
}

void runProgram(char* argv[])
{
    readFromFile();

    if (std::string(argv[2]) == "sec")
    {
        auto start = std::chrono::high_resolution_clock::now();
        sequential();
        auto end = std::chrono::high_resolution_clock::now();
        duration = end - start;
        writeMatrixToFile("result-sec.txt");
        return;
    }
    auto start = std::chrono::high_resolution_clock::now();
    multiThread(atoi(argv[3]));
    auto end = std::chrono::high_resolution_clock::now();
    duration = end - start;
    writeMatrixToFile("result-row.txt");
    compareResults();
}

void compareResults()
{
    std::ifstream finSec("result-sec.txt");
    std::ifstream finPar("result-row.txt");
    int tempSec, tempPar;
    for (int i = 0; i < rowSize; i++) {
        for (int j = 0; j < columnSize; j++) {
            finSec >> tempSec;
            finPar >> tempPar;
            if (tempSec != tempPar) {
                throw "Wrong result";
            }
        }
    }
    finSec.close();
    finPar.close();
}