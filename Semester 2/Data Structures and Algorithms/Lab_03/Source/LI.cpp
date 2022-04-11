#include <exception>
#include <iostream>
#include "LI.h"
#include "IteratorLI.h"

// Teta(1)
int LI::Aloca()
{
	int index = this->FirstFree;
	this->FirstFree = this->Next[this->FirstFree];
	return index;
}

// Teta(1)
void LI::Dealoca(int Index)
{
	this->Next[Index] = this->FirstFree;
	this->FirstFree = Index;
}

// Teta(2*n): n - vechiul numar de elemente
void LI::Resize()
{
	int newSize = 2 * this->Size;

	if (newSize < this->Size)
	{
		return; // Overflow
	}

	TElem* newElems = new TElem[newSize];
	int* newNext = new int[newSize];

	for (int index = 0; index < this->Size; index = index + 1)
	{
		newElems[index] = this->Elems[index];
		newNext[index] = this->Next[index];
	}

	for (int index = this->Size; index < newSize - 1; index = index + 1)
	{
		newNext[index] = index + 1;
	}
	newNext[newSize - 1] = -1;

	delete[] this->Elems;
	delete[] this->Next;

	this->FirstFree = this->Size;
	this->Size = newSize;
	this->Elems = newElems;
	this->Next = newNext;
}

// Teta(1)
int LI::CreateNode(TElem elem)
{
	if (-1 == this->FirstFree)
	{
		this->Resize();
	}
	int index = this->Aloca();

	this->Elems[index] = elem;
	this->Next[index] = -1;
	return index;
}

// Teta(n): n - dimensiunea initiala
LI::LI() {
	this->Count = 0;
	this->Size = SIZE;

	this->Elems = new TElem[SIZE];
	this->Next = new int[SIZE];

	for (int index = 0; index < SIZE - 1; index = index + 1)
	{
		this->Next[index] = index + 1;
	}

	this->Next[SIZE - 1] = -1;
	this->First = -1;
	this->FirstFree = 0;
}

// Teta(1)
int LI::dim() const {
	return this->Count;
}

// Teta(1)
bool LI::vida() const {
	return (0 == this->Count);
}

// O(n): n - numarul de elemente
TElem LI::element(int i) const {
	/* de adaugat */
	if (i < 0 || i >= this->Count)
	{
		throw std::exception();
	}
	int current = 0, auxPoz = this->First;

	while (current < i)
	{
		auxPoz = this->Next[auxPoz];
		current = current + 1;
	}
	return this->Elems[auxPoz];
}

// O(n): n - numarul de elemente
TElem LI::modifica(int i, TElem e) {
	/* de adaugat */
	if (i < 0 || i >= this->Count)
	{
		throw std::exception();
	}
	TElem toReturn{};
	int current = 0, auxPoz = this->First;

	while (current < i)
	{
		auxPoz = this->Next[auxPoz];
		current = current + 1;
	}
	toReturn = this->Elems[auxPoz];
	this->Elems[auxPoz] = e;
	return toReturn;
}

// Teta(n): n - numarul de elemente din lista
void LI::adaugaSfarsit(TElem e) {
	/* de adaugat */
	int poz = this->CreateNode(e);
	int auxPoz = this->First;

	this->Count = this->Count + 1;

	if (this->First == -1)
	{
		this->First = poz;
		return;
	}

	while (this->Next[auxPoz] != -1)
	{
		auxPoz = this->Next[auxPoz];
	}
	this->Next[auxPoz] = poz;
}

// O(n): n - numarul de elemente din lista
void LI::adauga(int i, TElem e) {
	/* de adaugat */
	if (i < 0 || i >= this->Count)
	{
		throw std::exception();
	}
	int poz = this->CreateNode(e);
	int auxPoz = this->First;
	int current = 0;

	this->Count = this->Count + 1;

	if (this->First == -1)
	{
		this->First = poz;
		return;
	}
	if (0 == i)
	{
		this->Next[poz] = this->First;
		this->First = poz;
		return;
	}
	while (current < i - 1)
	{
		auxPoz = this->Next[auxPoz];
		current = current + 1;
	}
	this->Next[poz] = this->Next[auxPoz];
	this->Next[auxPoz] = poz;
}

// O(n): n - numarul de elemente din lista
TElem LI::sterge(int i) {
	/* de adaugat */
	if (i < 0 || i >= this->Count)
	{
		throw std::exception();
	}
	TElem toReturn{};
	int current = 0, auxPoz = this->First;

	this->Count = this->Count - 1;

	if (0 == i)
	{
		toReturn = this->Elems[this->First];
		i = this->First;
		this->First = this->Next[this->First];
		goto deleteElem;
	}

	while (current < i - 1)
	{
		auxPoz = this->Next[auxPoz];
		current = current + 1;
	}
	toReturn = this->Elems[this->Next[auxPoz]];
	this->Next[auxPoz] = this->Next[this->Next[auxPoz]];

deleteElem:
	this->Dealoca(i);
	return toReturn;
}

// O(n): n - numarul de elemente din lista
int LI::cauta(TElem e) const{
	/* de adaugat */
	int auxPoz = this->First;
	int cont = 0;

	while (auxPoz != -1)
	{
		if (this->Elems[auxPoz] == e)
		{
			return cont;
		}
		cont = cont + 1;
		auxPoz = this->Next[auxPoz];
	}

	return -1;
}

// Teta(n): n - numarul de elemente
int LI::ultimulIndex(TElem elem) const
{
	int poz = -1, cont = 0;
	int auxPoz = First;

	while (auxPoz != -1)
	{
		if (Elems[auxPoz] == elem)
		{
			poz = cont;
		}
		cont = cont + 1;
		auxPoz = this->Next[auxPoz];
	}

	return poz;
}

// Teta(1)
IteratorLI LI::iterator() const {
	return  IteratorLI(*this);
}

// Teta(1)
LI::~LI() {
	/* de adaugat */
	delete[] this->Elems;
	delete[] this->Next;
}
