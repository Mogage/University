#pragma once
#include <iostream>
#include <mutex>
#include <vector>
#include <map>

class LinkedList
{
private:
    struct Node
    {
        int id;
        int score;
        Node* next;
        std::mutex mutex;
    };

    Node* head;
    Node* tail;
    int size;
    std::vector<int> blackList;
    std::map<int, std::mutex> appearnceList;
    std::mutex blackListMutex;
    std::mutex mutex;

    void remove(int);
    bool tryAddSameId(Node*);
    void addNode(Node*);
public:
    LinkedList();
    ~LinkedList();
    void add(int, int);
    void print(std::ostream&);
    int getSize();
};
