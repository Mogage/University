#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <thread>
#include <chrono>
#include <random>
#include <barrier>
#include <Windows.h>

#include "LinkedList.h"
#include "CustomQueue.h"

LinkedList list;
int numberOfCountries = 5;
int numberOfProblems = 10;

void reader(std::barrier<>&, CustomQueue&, int, int);

void worker(std::barrier<>&, CustomQueue&);

void sequential();

void parallel(char* []);

void createFiles();

void writeFile(std::string);

void verifyResuls();

int main(int argc, char* argv[])
{
    if (atoi(argv[1]) == 1)
    {
        createFiles();
        return 0;
    }   

    if (atoi(argv[1]) == 2)
    {
        auto begin = std::chrono::steady_clock::now();
        sequential();
        auto end = std::chrono::steady_clock::now();
        std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
        return 0;
    }
    auto begin = std::chrono::steady_clock::now();
    parallel(argv);
    auto end = std::chrono::steady_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();

    verifyResuls();

    return 0;
}

void sequential()
{
    for (int countryIndex = 0; countryIndex < numberOfCountries; countryIndex++)
    {
        for (int problemIndex = 0; problemIndex < numberOfProblems; problemIndex++)
        {
            std::ifstream in("Data\\C" + std::to_string(countryIndex) + "_P" + std::to_string(problemIndex) + ".txt");
            int id, score;
            while (in >> id >> score)
            {
                list.add(id, score);
            }
        }
    }
    writeFile("Output-sec.txt");
}

void parallel(char* argv[])
{
    int numberOfReaders = atoi(argv[2]);
    int numberOfWorkers = atoi(argv[3]);
    CustomQueue queue(numberOfReaders);
    std::vector<std::thread> readerThreads(numberOfReaders);
    std::vector<std::thread> workerThreads(numberOfWorkers);
    std::barrier barrier(numberOfWorkers + 1);
    int batchSize = numberOfCountries * numberOfProblems / numberOfReaders;
    int start = 0;
    int end;

    for (int i = 0; i < numberOfReaders; i++)
    {
        end = start + batchSize;
        readerThreads[i] = std::thread(reader, std::ref(barrier), std::ref(queue), start, end);
        start = end;
    }
    for (int i = 0; i < numberOfWorkers; i++)
    {
        workerThreads[i] = std::thread(worker, std::ref(barrier), std::ref(queue));
    }

    for (int i = 0; i < numberOfWorkers; i++)
    {
        workerThreads[i].join();
    }
    for (int i = 0; i < numberOfReaders; i++)
    {
        readerThreads[i].join();
    }
}

void reader(std::barrier<>& barrier, CustomQueue& queue, int start, int end)
{
    int countryIndex;
    int problemIndex;
    int id, score;

    for (int index = start; index < end; index++)
    {
        countryIndex = index / numberOfProblems;
        problemIndex = index % numberOfProblems;
        std::ifstream in("Data\\C" + std::to_string(countryIndex) + "_P" + std::to_string(problemIndex) + ".txt");
        while (in >> id >> score)
        {
            queue.push(id, score);
        }
    }
    queue.decrementReaders();
    if (start == 0)
    {
        barrier.arrive_and_wait();
        writeFile("Output-par.txt");
    }
}

void worker(std::barrier<>& barrier, CustomQueue& queue)
{
    while (queue.hasData())
    {
        auto data = queue.pop();
        if (data.first == -1)
        {
            Sleep(100);
            continue;
        }
        list.add(data.first, data.second);
    }
    barrier.arrive_and_drop();
}

void createFiles()
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> participantsDistribution(80, 100);
    std::uniform_int_distribution<int> idDistribution(1, 500);
    std::uniform_int_distribution<int> scoreDistribution(-1, 100);

    int numberOfParticipants;
    int score;

    for (int countryIndex = 0; countryIndex < numberOfCountries; countryIndex++)
    {
        for (int problemIndex = 0; problemIndex < numberOfProblems; problemIndex++)
        {
            std::ofstream out("Data\\C" + std::to_string(countryIndex) + "_P" + std::to_string(problemIndex) + ".txt");
            numberOfParticipants = participantsDistribution(gen);
            for (int i = 0; i < numberOfParticipants; i++)
            {
                score = scoreDistribution(gen);
                if (score == 0)
                {
                    continue;
                }
                out << idDistribution(gen) << " " << score << "\n";
            }
        }
    }
}

void writeFile(std::string fileName)
{
    std::ofstream out(fileName);
    list.print(out);
}

void verifyResuls()
{
    std::ifstream inSec("Output-sec.txt");
    std::ifstream inPar("Output-par.txt");
    std::string lineSec;
    std::string linePar;
    while (std::getline(inSec, lineSec))
    {
        std::getline(inPar, linePar);
        if (lineSec != linePar)
        {
            std::cout << "Results are not the same\n";
            return;
        }
    }
    std::cout << "Results are the same\n";
}