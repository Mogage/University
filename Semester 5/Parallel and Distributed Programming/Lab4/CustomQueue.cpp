#include "CustomQueue.h"
#include <chrono>
#include <iostream>

CustomQueue::CustomQueue() : sem(0)
{
    this->numberOfReaders = 1;
}

CustomQueue::CustomQueue(int numberOfReaders) : sem(0)
{
    this->numberOfReaders = numberOfReaders;
}

void CustomQueue::decrementReaders()
{
    std::lock_guard<std::mutex> lock(mutex);
    numberOfReaders--;
}

bool CustomQueue::hasData()
{
    std::lock_guard<std::mutex> lock(mutex);
    if (numberOfReaders == 0)
    {
        return !queue.empty();
    }
    return true;
}

void CustomQueue::push(int id, int score)
{
    std::lock_guard<std::mutex> lock(mutex);
    queue.push(std::make_pair(id, score));
    sem.release();
}

std::pair<int, int> CustomQueue::pop()
{
    std::lock_guard<std::mutex> lock(mutex);
    if (sem.try_acquire())
    {
        std::pair<int, int> value = queue.front();
        queue.pop();
        return value;
    }
    return std::make_pair(-1, -1);
}

int CustomQueue::size()
{
    return queue.size();
}