#include "Colectie.h"
#include "IteratorColectie.h"
#include <exception>
#include <iostream>
#include <stdlib.h>

using namespace std;

/*  Complexitate Constructor: Teta(1)*/
Colectie::Colectie() {
	/* de adaugat */
	this->Count = 0;
	this->Size = 10000;
	this->Array = new ELEM[Size];
}

/*  Complexitate adauga: O(n), n - numarul de elemente distincte din colectie*/
void Colectie::adauga(TElem elem) {
	/* de adaugat */
	int index;
	bool found = false;
	// Cazul in care s-a umplut colectia
	if (this->Count >= this->Size - 1)
	{
		int newSize = this->Size * 2;
		if (newSize < this->Size) // Overflow
		{
			return;
		}

		ELEM* newArray = new ELEM[newSize];

		memcpy(newArray, this->Array, this->Size * sizeof(ELEM));

		this->Size = newSize;
		delete[]this->Array;
		this->Array = newArray;
	}
	for (index = 0; index < this->Count; index = index + 1)
	{
		if (this->Array[index].Value == elem)
		{
			this->Array[index].Frequency = this->Array[index].Frequency + 1;
			found = true;
			break;
		}
	}
	if (false == found)
	{
		this->Array[index].Value = elem;
		this->Array[index].Frequency = 1;
		this->Count = this->Count + 1;
	}
}

/*  Complexitate sterge: O(n), n - numarul de elemente distincte din colectie*/
bool Colectie::sterge(TElem elem) {
	/* de adaugat */
	int index;
	bool found = false;
	for (index = 0; index < this->Count; index = index + 1)
	{
		if (this->Array[index].Value == elem)
		{
			found = true;
			break;
		}
	}
	if (true == found)
	{
		if (this->Array[index].Frequency > 1)
		{
			this->Array[index].Frequency = this->Array[index].Frequency - 1;
		}
		else
		{
			this->Count = this->Count - 1;
			for (; index < this->Count; index = index + 1)
			{
				this->Array[index] = this->Array[index + 1];
			}
			this->Array[this->Count].Frequency = 0;
		}
		return true;
	}
	return false;
}

/*  Complexitate cauta: O(n), n - numarul de elemente distincte din colectie*/
bool Colectie::cauta(TElem elem) const {
	/* de adaugat */
	for (int index = 0; index < this->Count; index = index + 1)
	{
		if (this->Array[index].Value == elem)
		{
			return true;
		}
	}
	return false;
}

/*  Complexitate nrAparitii: O(n), n - numarul de elemente distincte din colectie*/
int Colectie::nrAparitii(TElem elem) const {
	/* de adaugat */
	for (int index = 0; index < this->Count; index = index + 1)
	{
		if (this->Array[index].Value == elem)
		{
			return this->Array[index].Frequency;
		}
	}
	return 0;
}

/*  Complexitate dim: Teta(n), n - numarul de elemente distincte din colectie*/
int Colectie::dim() const {
	/* de adaugat */
	int total = 0;
	for (int index = 0; index < this->Count; index = index + 1)
	{
		total = total + this->Array[index].Frequency;
	}
	return total;
}

/*  Complexitate vida: teta(1) */
bool Colectie::vida() const {
	/* de adaugat */
	if (0 == this->Count)
	{
		return true;
	}
	return false;
}

IteratorColectie Colectie::iterator() const {
	return  IteratorColectie(*this);
}

/* Complexitate Destructor: teta(1) */
Colectie::~Colectie() {
	/* de adaugat */
	delete[]this->Array;
}


