#include "CustomQueue.h"

#include <iostream>
#include <string>

CustomQueue::CustomQueue()
{
    maxSize = 50;
    numberOfReaders = 1;
}

CustomQueue::CustomQueue(int numberOfReaders)
{
    this->maxSize = 50;
    this->numberOfReaders = numberOfReaders;
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
    std::unique_lock<std::mutex> lock(mutex);
    if (numberOfReaders == 0)
    {
        return !queue.empty();
    }
    return true;
}

void CustomQueue::push(int id, int score)
{
	std::unique_lock<std::mutex> lock(mutex);
	cvPush.wait(lock, [this]() {return queue.size() < maxSize; });
	queue.push(std::make_pair(id, score));
	cvPop.notify_all();
}

std::pair<int, int> CustomQueue::pop()
{
	std::unique_lock<std::mutex> lock(mutex);
	cvPop.wait(lock, [this]() {return numberOfReaders != 0 || !queue.empty(); });
    if (queue.empty())
    {
		return std::make_pair(-1, -1);
	}

	std::pair<int, int> result = queue.front();
	queue.pop();
	cvPush.notify_one();
	return result;
}

int CustomQueue::size()
{
    return queue.size();
}
