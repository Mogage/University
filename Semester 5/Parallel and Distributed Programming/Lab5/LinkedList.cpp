#include "LinkedList.h"

#include <string>

LinkedList::LinkedList()
{
	head = new Node();
	tail = new Node();
	head->next = tail;
	tail->next = nullptr;
	tail->score = INT_MIN;
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

	this->mutex.lock();

	if (std::find(blackList.begin(), blackList.end(), id) != blackList.end())
	{
		delete newNode;
		this->mutex.unlock();
		return;
	}

	if (score == -1)
	{
		blackList.push_back(id);
		this->remove(id);
		delete newNode;
		this->mutex.unlock();
		return;
	}

	if (tryAddSameId(newNode))
	{
		this->mutex.unlock();
		return;
	}
	this->mutex.unlock();
	addNode(newNode);
	size++;
}

void LinkedList::addNode(Node* newNode)
{
	Node* previous = nullptr;
	Node* current = head;
	
	while (current != tail)
	{
		previous = current;
		current = current->next;

		std::unique_lock<std::mutex> lock2(current->mutex);
		std::unique_lock<std::mutex> lock1(previous->mutex);

		if (current->score < newNode->score || current->score == newNode->score && current->id < newNode->id)
		{
			newNode->next = current;
			previous->next = newNode;
			break;
		}

		lock1.unlock();
		lock2.unlock();
	}
}

bool LinkedList::tryAddSameId(Node* newNode)
{
	Node* previous = nullptr;
	Node* current = head;
	while (current != tail)
	{
		previous = current;
		current = current->next;

		std::unique_lock<std::mutex> lock2(current->mutex);
		std::unique_lock<std::mutex> lock1(previous->mutex);

		if (current->id == newNode->id) 
		{
			current->score += newNode->score;
			previous->next = current->next;

			lock1.unlock();
			lock2.unlock();
			
			delete newNode;
			addNode(current);
			return true;
		}

		lock1.unlock();
		lock2.unlock();
	}
	return false;
}

void LinkedList::remove(int id)
{
	Node* previous = nullptr;
	Node* current = head;
	while (current != tail)
	{
		previous = current;
		current = current->next;

		std::unique_lock<std::mutex> lock2(current->mutex);
		std::unique_lock<std::mutex> lock1(previous->mutex);

		if (current->id == id) {
			previous->next = current->next;

			lock1.unlock();
			lock2.unlock();

			delete current;
			return;
		}
		lock1.unlock();
		lock2.unlock();
	}
}

void LinkedList::print(std::ostream& out)
{
	Node* current = head->next;
	while (current != tail)
	{
		out << current->id << " " << current->score << std::endl;
		current = current->next;
	}
}

int LinkedList::getSize()
{
	return size;
}