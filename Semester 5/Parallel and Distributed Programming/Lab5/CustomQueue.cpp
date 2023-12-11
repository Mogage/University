#include "CustomQueue.h"

CustomQueue::CustomQueue()
{
    maxSize = 50;
    numberOfReaders = 1;
}

CustomQueue::CustomQueue(int maxSize, int numberOfReaders)
{
    this->maxSize = maxSize;
    this->numberOfReaders = numberOfReaders;
}

void CustomQueue::decrementReaders()
{
    numberOfReaders--;
}

bool CustomQueue::hasData()
{
    return !queue.empty();
}

void CustomQueue::push(int id, int score)
{
    std::unique_lock<std::mutex> lock(mutex);
    cv.wait(lock, [this]() {return queue.size() < maxSize; });
    queue.push(std::make_pair(id, score));
    cv.notify_all();
}

std::pair<int, int> CustomQueue::pop()
{
    std::unique_lock<std::mutex> lock(mutex);
    cv.wait(lock, [this]() {return !queue.empty(); });
    std::pair<int, int> pair = queue.front();
    queue.pop();
    cv.notify_all();
    return pair;
}

int CustomQueue::size()
{
    return queue.size();
}
