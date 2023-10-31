#include <iostream>
#include <fstream>
#include <thread>
#include <chrono>
#include <string>
#include <vector>
#include <random>

const std::string fileName = "data.txt";
std::chrono::duration<double, std::milli> duration;

const int rowSize = 10000;
const int columnSize = 10;
const int smallSize = 5;

//int resultMatrix[rowSize][columnSize];
//int bigMatrix[rowSize][columnSize];
//int smallMatrix[smallSize][smallSize];

std::vector<std::vector<int>> resultMatrix(rowSize, std::vector<int>(columnSize));
std::vector<std::vector<int>> bigMatrix(rowSize, std::vector<int>(columnSize));
std::vector<std::vector<int>> smallMatrix(smallSize, std::vector<int>(smallSize));

void createFile() 
{
    std::ofstream fout(fileName);
    for (int i = 0; i < rowSize; i++) {
        for (int j = 0; j < columnSize; j++) {
            bigMatrix[i][j] = rand() % 101;
            fout << bigMatrix[i][j] << " ";
        }
        fout << '\n';
    }

    for (int i = 0; i < smallSize; i++) {
        for (int j = 0; j < smallSize; j++) {
            smallMatrix[i][j] = rand() % 2;
            fout << smallMatrix[i][j] << " ";
        }
        fout << '\n';
    }
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
}

void writeMatrixToFile(std::string filename)
{
    std::ofstream fout(filename);
    for (int i = 0; i < rowSize; i++) {
        for (int j = 0; j < columnSize; j++) {
            fout << resultMatrix[i][j] << " ";
        }
        fout << '\n';
    }
}

void createBorder() 
{
    for (int i = 1; i < rowSize + 1; i++) {
        bigMatrix[i][0] = bigMatrix[i][1];
        bigMatrix[i][columnSize + 1] = bigMatrix[i][columnSize];
    }
    for (int i = 1; i < columnSize + 1; i++) {
        bigMatrix[0][i] = bigMatrix[1][i];
        bigMatrix[rowSize + 1][i] = bigMatrix[rowSize][i];
    }
    bigMatrix[0][0] = bigMatrix[1][1];
    bigMatrix[0][columnSize + 1] = bigMatrix[1][columnSize];
    bigMatrix[rowSize + 1][0] = bigMatrix[rowSize][1];
    bigMatrix[rowSize + 1][columnSize + 1] = bigMatrix[rowSize][columnSize];
}

void sequential()
{
    auto startSequential = std::chrono::high_resolution_clock::now();
    int rowIndex;
    int columnIndex;
    int sum;
    for (int i = 0; i < rowSize; i++) {
        for (int j = 0; j < columnSize; j++) {
            sum = 0;
            for (int index1 = 0; index1 < smallSize; index1++) {
                for (int index2 = 0; index2 < smallSize; index2++) {
                    if (i - smallSize / 2 + index1 < 0)
                        rowIndex = 0;
                    else if (i - smallSize / 2 + index1 >= rowSize)
                        rowIndex = rowSize - 1;
                    else
                        rowIndex = i - smallSize / 2 + index1;
                    if (j - smallSize / 2 + index2 < 0)
                        columnIndex = 0;
                    else if (j - smallSize / 2 + index2 >= columnSize)
                        columnIndex = columnSize - 1;
                    else
                        columnIndex = j - smallSize / 2 + index2;
                    sum += bigMatrix[rowIndex][columnIndex] * smallMatrix[index1][index2];
                }
            }
            resultMatrix[i][j] = sum;
        }
    }
    auto endSequential = std::chrono::high_resolution_clock::now();
    duration = endSequential - startSequential;
}

void runOnRows(int start, int end)
{
    int rowIndex;
    int columnIndex;
    int sum;
    for (int i = start; i < end; i++) {
        for (int j = 0; j < columnSize; j++) {
            sum = 0;
            for (int index1 = 0; index1 < smallSize; index1++) {
                for (int index2 = 0; index2 < smallSize; index2++) {
                    if (i - smallSize / 2 + index1 < 0)
                        rowIndex = 0;
                    else if (i - smallSize / 2 + index1 >= rowSize)
                        rowIndex = rowSize - 1;
                    else
                        rowIndex = i - smallSize / 2 + index1;
                    if (j - smallSize / 2 + index2 < 0)
                        columnIndex = 0;
                    else if (j - smallSize / 2 + index2 >= columnSize)
                        columnIndex = columnSize - 1;
                    else
                        columnIndex = j - smallSize / 2 + index2;
                    sum += bigMatrix[rowIndex][columnIndex] * smallMatrix[index1][index2];
                }
            }
            resultMatrix[i][j] = sum;
        }
    }
}

void runOnColumns(int start, int end)
{
    int rowIndex;
    int columnIndex;
    int sum;
    for (int i = 0; i < rowSize; i++) {
        for (int j = start; j < end; j++) {
            sum = 0;
            for (int index1 = 0; index1 < smallSize; index1++) {
                for (int index2 = 0; index2 < smallSize; index2++) {
                    if (i - smallSize / 2 + index1 < 0)
                        rowIndex = 0;
                    else if (i - smallSize / 2 + index1 >= rowSize)
                        rowIndex = rowSize - 1;
                    else
                        rowIndex = i - smallSize / 2 + index1;
                    if (j - smallSize / 2 + index2 < 0)
                        columnIndex = 0;
                    else if (j - smallSize / 2 + index2 >= columnSize)
                        columnIndex = columnSize - 1;
                    else
                        columnIndex = j - smallSize / 2 + index2;
                    sum += bigMatrix[rowIndex][columnIndex] * smallMatrix[index1][index2];
                }
            }
            resultMatrix[i][j] = sum;
        }
    }
}

void runOnSubMatrix(int start, int end)
{
    int rowIndex;
    int columnIndex;
    int i;
    int j;
    int sum;
    for (int aux = start; aux < end; aux++) {
        sum = 0;
        i = aux / columnSize;
        j = aux % columnSize;
        for (int index1 = 0; index1 < smallSize; index1++) {
            for (int index2 = 0; index2 < smallSize; index2++) {
                if (i - smallSize / 2 + index1 < 0)
                    rowIndex = 0;
                else if (i - smallSize / 2 + index1 >= rowSize)
                    rowIndex = rowSize - 1;
                else
                    rowIndex = i - smallSize / 2 + index1;
                if (j - smallSize / 2 + index2 < 0)
                    columnIndex = 0;
                else if (j - smallSize / 2 + index2 >= columnSize)
                    columnIndex = columnSize - 1;
                else
                    columnIndex = j - smallSize / 2 + index2;
                sum += bigMatrix[rowIndex][columnIndex] * smallMatrix[index1][index2];
            }
        }
        resultMatrix[i][j] = sum;
    }
}

void multiThread(void (*function)(int, int), int threadCount, int batchSize, int batchReminder, int startValue)
{
    int start = startValue;
    int end;
    std::vector<std::thread> threads(threadCount);

    auto startParallel = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < threadCount; i++) {
        end = start + batchSize;
        if (batchReminder > 0) {
            end++;
            batchReminder--;
        }
        threads[i] = std::thread(function, start, end);
        start = end;
    }

    for (int i = 0; i < threadCount; i++) {
        threads[i].join();
    }

    auto endParallel = std::chrono::high_resolution_clock::now();
    duration = endParallel - startParallel;
}

void multiThreadRows(int threadCount = 4)
{
    int batchSize = rowSize / threadCount;
    int batchReminder = rowSize % threadCount;
    multiThread(runOnRows, threadCount, batchSize, batchReminder, 0);
}   

void multiThreadColumns(int threadCount = 4)
{
    int batchSize = columnSize / threadCount;
    int batchReminder = columnSize % threadCount;
    multiThread(runOnColumns, threadCount, batchSize, batchReminder, 0);
}

void multiThreadSubMatrix(int threadCount = 4)
{
    int batchSize = (rowSize * columnSize) / threadCount;
    int batchReminder = (rowSize * columnSize) % threadCount;
    multiThread(runOnSubMatrix, threadCount, batchSize, batchReminder, 0);
}

void compareResults(std::string fileName)
{
    std::ifstream finSec("output-sec.txt");
    std::ifstream finPar("output-" + fileName + ".txt");
    int tempSec, tempPar;
    bool isSame = true;
    for (int i = 0; i < rowSize; i++) {
        for (int j = 0; j < columnSize; j++) {
            finSec >> tempSec;
            finPar >> tempPar;
            if (tempSec != tempPar) {
                isSame = false;
                throw "Wrong result";
            }
        }
    }
}

void runProgram(char* argv[])
{
    int threadCount = atoi(argv[1]);

    readFromFile();
    //createBorder();

    std::string outputFileName = std::string(argv[2]);

    if (outputFileName == "sec")
    {
        sequential();
        writeMatrixToFile("output-" + outputFileName + ".txt");
    }
    else
    {
        if (outputFileName == "row")
        {
            multiThreadRows(threadCount);
        }
        else if (outputFileName == "col")
        {
            multiThreadColumns(threadCount);
        }
        else
        {
            multiThreadSubMatrix(threadCount);
        }
        writeMatrixToFile("output-" + outputFileName + ".txt");
        compareResults(outputFileName);
    }
}

int main(int argc, char* argv[])
{
    //createFile();
    runProgram(argv);
    std::cout << duration.count();

    return 0;
}
