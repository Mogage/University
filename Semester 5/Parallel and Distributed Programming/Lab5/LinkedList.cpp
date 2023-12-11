#include "LinkedList.h"

LinkedList::LinkedList()
{
    head = nullptr;
    tail = nullptr;
    size = 0;
}

LinkedList::~LinkedList()
{
    Node* current = head;
    while (current != nullptr)
    {
        Node* next = current->next;
        delete current;
        current = next;
    }
}

void LinkedList::add(int id, int score)
{
    Node* newNode = new Node();
    newNode->id = id;
    newNode->score = score;
    newNode->next = nullptr;
    newNode->previous = nullptr;

    std::unique_lock<std::mutex> lock(mutex);
    if (head == nullptr)
    {
        head = newNode;
        tail = newNode;
    }
    else
    {
        addNode(newNode, head);
    }
    size++;
}

