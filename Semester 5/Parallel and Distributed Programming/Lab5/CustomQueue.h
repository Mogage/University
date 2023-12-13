#pragma once
#include <queue>
#include <mutex>

class CustomQueue
{
private:
    std::queue<std::pair<int, int>> queue;
    int maxSize;
    int numberOfReaders;
    std::mutex mutex;
    std::condition_variable cvPop;
    std::condition_variable cvPush;
public:
    CustomQueue();
    CustomQueue(int numberOfReaders);
    CustomQueue(int maxSize, int numberOfReaders);
    void decrementReaders();
    bool hasData();
    void push(int id, int score);
    std::pair<int, int> pop();
    int size();
};
