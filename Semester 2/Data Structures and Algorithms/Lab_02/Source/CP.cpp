
#include "CP.h"
#include <exception>
#include <stdio.h>

// Teta(1)
CP::CP(Relatie r) {
	this->Front = NULL;
	this->Compare = r;
}

// O(n) ; n - numarul de elemente din lista
void CP::adauga(TElem e, TPrioritate p) {
	PList toAdd = new List(Element(e, p), NULL);
	PList index = NULL;

	// Daca nu exista niciun element in lista sau elementul de adaugat are cea mai mare prioritate
	if (NULL == Front || true == this->Compare(p, Front->Value.second))
	{
		toAdd->Next = Front;
		Front = toAdd;
		return;
	}

	// altfel il parcurg pana elementul curent 
	index = Front;
	while (NULL != index->Next && true == this->Compare(index->Next->Value.second, p))
	{
		index = index->Next;
	}
	toAdd->Next = index->Next;
	index->Next = toAdd;
}

// Teta(1)
Element CP::element() const {
	if (NULL == this->Front)
	{
		throw std::exception();
	}
	return Element(Front->Value);  // copy constructor
;
}

// Teta(1)
Element CP::sterge() {
	if (NULL == this->Front)
	{
		throw std::exception();
	}
	Element deleted(Front->Value);
	PList toDelete = Front;

	Front = Front->Next;
	delete toDelete;

	return deleted;
}

// Teta(1)
bool CP::vida() const {
	return (NULL == this->Front);
}

// O(n * m) - m - numarul de elemente din cp, n - numarul de elemente din coada curenta
void CP::imbinare(CP& cp)
{
	Element aux{ };

	while (false == cp.vida())
	{
		aux = cp.sterge();
		this->adauga(aux.first, aux.second);
	}
	
}

// Teta(n) ; n - numarul de elemente din lista
CP::~CP() {
	PList aux{ NULL };
	while (NULL != Front)
	{
		aux = Front;
		Front = Front->Next;
		delete aux;
	}
};

// Teta(1)
List::List(Element Elem, PList NextNode)
{
	this->Value = Elem;
	this->Next = NextNode;
}
