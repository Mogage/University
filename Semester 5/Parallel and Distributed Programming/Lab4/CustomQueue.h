#pragma once
#include <queue>
#include <semaphore>
#include <mutex>

class CustomQueue
{
private:
    std::queue<std::pair<int, int>> queue;
    int numberOfReaders;
    std::mutex mutex;
    std::counting_semaphore<10000> sem;
public:
    CustomQueue();
    CustomQueue(int numberOfReaders);

    void decrementReaders();

    bool hasData();

    void push(int id, int score);

    std::pair<int, int> pop();

    int size();
};


