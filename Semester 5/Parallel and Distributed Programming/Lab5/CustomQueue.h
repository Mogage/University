#pragma once
#include <queue>
#include <mutex>

// implement a custom queue with a wait notify system using condition variables
class CustomQueue
{
private:
    std::queue<std::pair<int, int>> queue;
    int maxSize;
    int numberOfReaders;
    std::mutex mutex;
    std::condition_variable cv;
public:
    CustomQueue();
    CustomQueue(int maxSize, int numberOfReaders);
    void decrementReaders();
    bool hasData();
    void push(int id, int score);
    std::pair<int, int> pop();
    int size();
};

//class CustomQueue
//{
//private:
//    std::queue<std::pair<int, int>> queue;
//    int numberOfReaders;
//    std::mutex mutex;
//public:
//    CustomQueue();
//    CustomQueue(int numberOfReaders);
//
//    void decrementReaders();
//
//    bool hasData();
//
//    void push(int id, int score);
//
//    std::pair<int, int> pop();
//
//    int size();
//};
//

