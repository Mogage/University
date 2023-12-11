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

bool LinkedList::verifyCurrentNode(Node* current)
{
    if (current == nullptr)
    {
        return false;
    }
    if (current->previous == nullptr)
    {
        return false;
    }
    if (current->score < current->previous->score)
    {
        return false;
    }
    if (current->score == current->previous->score && current->id < current->previous->id)
    {
        return false;
    }
    return true;
}

bool LinkedList::tryAddSameId(int id, int score)
{
    Node* current = head;
    while (current != nullptr)
    {
        if (current->id == id)
        {
            current->score += score;
            while (verifyCurrentNode(current))
            {
                if (current == tail)
                {
                    tail = current->previous;
                }
                current->previous->next = current->next;
                if (current->next != nullptr)
                {
                    current->next->previous = current->previous;
                }
                current->next = current->previous;
                current->previous = current->previous->previous;
                current->next->previous = current;
                if (current->previous != nullptr)
                {
                    current->previous->next = current;
                }
                else
                {
                    head = current;
                }
            }
            return true;
        }
        current = current->next;
    }
    return false;
}

void LinkedList::addNode(Node* newNode, Node* current)
{
    if (current == head)
    {
        newNode->next = head;
        head->previous = newNode;
        head = newNode;
    }
    else
    {
        newNode->next = current;
        newNode->previous = current->previous;
        current->previous->next = newNode;
        current->previous = newNode;
    }
    size++;
}

void LinkedList::add(int id, int score)
{
    mutex.lock();
    Node* newNode = new Node;
    newNode->id = id;
    newNode->score = score;
    newNode->next = nullptr;
    newNode->previous = nullptr;

    if (std::find(blackList.begin(), blackList.end(), id) != blackList.end())
    {
        mutex.unlock();
        return;
    }
    if (score == -1)
    {
        blackList.push_back(id);
        this->remove(id);
        mutex.unlock();
        return;
    }

    if (head == nullptr)
    {
        head = newNode;
        tail = newNode;
    }
    else
    {
        if (tryAddSameId(id, score))
        {
            mutex.unlock();
            return;
        }
        Node* current = head;
        while (current != nullptr)
        {
            if (current->score < score || current->score == score && current->id < id)
            {
                addNode(newNode, current);
                mutex.unlock();
                return;
            }
            current = current->next;
        }
        tail->next = newNode;
        newNode->previous = tail;
        tail = newNode;
    }
    size++;
    mutex.unlock();
}

void LinkedList::remove(int id)
{
    Node* current = head;

    if (size == 1)
    {
        if (current->id == id)
        {
            head = nullptr;
            tail = nullptr;
            delete current;
            size--;
        }
        return;
    }

    while (current != nullptr)
    {
        if (current->id == id)
        {
            if (current == head)
            {
                head = head->next;
                head->previous = nullptr;
            }
            else if (current == tail)
            {
                tail = tail->previous;
                tail->next = nullptr;
            }
            else
            {
                current->previous->next = current->next;
                current->next->previous = current->previous;
            }
            delete current;
            size--;
            break;
        }
        current = current->next;
    }
}

void LinkedList::print(std::ostream& outStream)
{
    Node* current = head;
    //outStream << "Black list: " << blackList.size() << "\n";
    //for (int i = 0; i < blackList.size(); i++)
    //{
    //    outStream << blackList[i] << ' ';
    //}
    //outStream << "\nList: " << size << "\n";
    while (current != nullptr)
    {
        outStream << current->id << " " << current->score << '\n';
        current = current->next;
    }
}

int LinkedList::getSize()
{
    return size;
}
