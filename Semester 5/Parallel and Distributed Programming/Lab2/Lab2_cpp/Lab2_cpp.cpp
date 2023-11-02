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

std::vector<std::vector<int>> createBuffer(int, int, int);
int computeSubMatrix(int, int, int, int, std::vector<std::vector<int>>);
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
        buffer[i].assign(bigMatrix[0].begin(), bigMatrix[0].end());
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
            buffer[index1].assign(buffer[index1 + 1].begin(), buffer[index1 + 1].end());
        }
        buffer[noOfRowsBuffer - 1].assign(bigMatrix[i + 1].begin(), bigMatrix[i + 1].end());
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

std::vector<std::vector<int>> createBuffer(int halfSmallSize, int noOfRowsBuffer, int start)
{
    std::vector<std::vector<int>> buffer = std::vector<std::vector<int>>(noOfRowsBuffer, std::vector<int>(columnSize));
    for (int i = 0; i < noOfRowsBuffer; i++)
    {
        if (i + start - halfSmallSize < 0 || i + start - halfSmallSize >= rowSize)
        {
            buffer[i].assign(bigMatrix[0].begin(), bigMatrix[0].end());
            continue;
        }
        buffer[i].assign(bigMatrix[i + start - halfSmallSize].begin(), bigMatrix[i + start - halfSmallSize].end());
    }
    return buffer;
}

int computeSubMatrix(int row, int col, int halfSmallSize, int noOfRowsBuffer, std::vector<std::vector<int>> buffer)
{
    int rowIndex;
    int columnIndex;
    int sum = 0;
    for (int index1 = 0; index1 < smallSize; index1++)
    {
        for (int index2 = 0; index2 < smallSize; index2++)
        {
            rowIndex = std::min(std::max(row - halfSmallSize + index1, 0), noOfRowsBuffer - 1);
            columnIndex = std::min(std::max(col - halfSmallSize + index2, 0), columnSize - 1);
            sum += buffer[rowIndex][columnIndex] * smallMatrix[index1][index2];
        }
    }
    return sum;
}

void runOnRows(std::barrier<>& barrier, int start, int end)
{
    int halfSmallSize = smallSize / 2;
    int noOfRowsBuffer = end - start + halfSmallSize * 2;
    std::vector<std::vector<int>> buffer = createBuffer(halfSmallSize, noOfRowsBuffer, start);
    if (end == rowSize)
    {
        noOfRowsBuffer -= halfSmallSize;
    }
    barrier.arrive_and_wait();
    for (int i = halfSmallSize; i < end - start + halfSmallSize; i++)
    {
        for (int j = 0; j < columnSize; j++)
        {
            bigMatrix[i + start - halfSmallSize][j] = computeSubMatrix(i, j, halfSmallSize, noOfRowsBuffer, buffer);
        }
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
    int threadCount = 4;

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
    multiThread(threadCount);
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