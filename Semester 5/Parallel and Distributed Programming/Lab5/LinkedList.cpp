#include "LinkedList.h"

#include <string>

LinkedList::LinkedList()
{
    head = new Node();
    tail = new Node();
    head->next = tail;
    tail->next = nullptr;
    tail->score = INT_MIN;
    head->score = INT_MAX;
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

    this->blackListMutex.lock();

    if (std::find(blackList.begin(), blackList.end(), id) != blackList.end())
    {
        this->blackListMutex.unlock();
        delete newNode;
        return;
    }

    if (score == -1)
    {
        blackList.push_back(id);
        blackListMutex.unlock();
        this->remove(id);
        delete newNode;
        return;
    }
    blackListMutex.unlock();

    if (tryAddSameId(newNode))
    {
        return;
    }
    addNode(newNode);
    size++;
}

void LinkedList::addNode(Node* newNode)
{
    Node* previous = nullptr;
    Node* current = head;

    current->mutex.lock();
    while (current != tail)
    {
        previous = current;
        current = current->next;
        current->mutex.lock();

        if (current->score < newNode->score || current->score == newNode->score && current->id < newNode->id)
        {
            newNode->next = current;
            previous->next = newNode;
            previous->mutex.unlock();
            break;
        }

        previous->mutex.unlock();
    }
    current->mutex.unlock();
}

bool LinkedList::tryAddSameId(Node* newNode)
{
    Node* previous = nullptr;
    Node* current = head;

    this->mutex.lock();
    current->mutex.lock();
    while (current != tail)
    {
        previous = current;
        current = current->next;
        current->mutex.lock();

        if (current->id == newNode->id)
        {
            current->score += newNode->score;
            current->next->mutex.lock();
            previous->next = current->next;

            previous->mutex.unlock();
            current->mutex.unlock();
            current->next->mutex.unlock();

            addNode(current);
            this->mutex.unlock();
            delete newNode;
            return true;
        }

        previous->mutex.unlock();
    }
    current->mutex.unlock();
    this->mutex.unlock();
    return false;
}

void LinkedList::remove(int id)
{
    Node* previous = nullptr;
    Node* current = head;

    current->mutex.lock();
    while (current != tail)
    {
        previous = current;
        current = current->next;
        current->mutex.lock();

        if (current->id == id) {
            current->next->mutex.lock();
            previous->next = current->next;

            previous->mutex.unlock();
            current->mutex.unlock();
            current->next->mutex.unlock();

            delete current;
            return;
        }
        previous->mutex.unlock();
    }
    current->mutex.unlock();
}

void LinkedList::print(std::ostream& out)
{
    Node* current = head->next;
    while (current != tail)
    {
        out << current->id << " " << current->score << "\n";
        current = current->next;
    }
}

int LinkedList::getSize()
{
    return size;
}