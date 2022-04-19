#include "Colectie.h"
#include "IteratorColectie.h"
#include <exception>
#include <iostream>

int Colectie::hash(TElem Element, int I) const
{
	return (int)(abs(Element) + 0.5 * I + 0.5 * I * I) % size;
}

Colectie::Colectie() {
	/* de adaugat */
	count = 0;
	size = INITIAL_SIZE;
	tabela = std::vector < std::pair < int, int > >(INITIAL_SIZE, { NULL_TELEM , NULL_TELEM});
}


void Colectie::adauga(TElem elem) {
	/* de adaugat */
	int index = 0, hashCode;
	do
	{
		hashCode = hash(elem, index);
		if (NULL_TELEM == tabela[hashCode].second)
		{
			tabela[hashCode] = { elem, 1 };
			break;
		}
		if (elem == tabela[hashCode].first)
		{
			tabela[hashCode].second = tabela[hashCode].second + 1;
			break;
		}
		index = index + 1;
	} while (index < size);
	if (index != size)
	{
		count = count + 1;
	}
}


bool Colectie::sterge(TElem elem) {
	/* de adaugat */
	int index = 0, hashCode = 0, deletedIndex, toDelete;
	do
	{
		// hashCode = hash(elem, index);
		if (elem == tabela[hashCode].first && NULL_TELEM != tabela[hashCode].second)
		{
			tabela[hashCode].second = tabela[hashCode].second - 1;
			count = count - 1;
			if (0 == tabela[hashCode].second)
			{
				break;
			}
			return true;
		}
		// index = index + 1;
		hashCode = hashCode + 1;
	} while (hashCode < size);
	if (hashCode == size)
	{
		return false;
	}
	deletedIndex = hashCode;
	index = hashCode;
	toDelete = hashCode;
	do
	{
		index = index + 1;
		if (index == size)
		{
			index = 0;
		}
		hashCode = hash(tabela[index].first, 0);
		if (deletedIndex < index)
		{
			if (hashCode > deletedIndex && hashCode <= index)
			{
				continue;
			}
			// if (hashCode <= deletedIndex)
			//if (index < hashCode)
			tabela[deletedIndex] = tabela[index];
			deletedIndex = index;
			toDelete = index;

		}
		else if(hashCode > index && hashCode <= deletedIndex)
		{
			tabela[deletedIndex] = tabela[index];
			deletedIndex = index;
			toDelete = index;
		}
	} while (NULL_TELEM != tabela[index].second);
	tabela[toDelete] = { NULL_TELEM, NULL_TELEM };
	return true;
}


bool Colectie::cauta(TElem elem) const {
	/* de adaugat */
	int index = 0;// , hashCode;
	do
	{
		if (elem == tabela[index].first && NULL_TELEM != tabela[index].second)
		{
			return true;
		}
		index = index + 1;
	} while (index < size);
	return false;
}

int Colectie::nrAparitii(TElem elem) const {
	/* de adaugat */
	int index = 0, hashCode;
	do
	{
		hashCode = hash(elem, index);
		if (elem == tabela[hashCode].first && NULL_TELEM != tabela[hashCode].second)
		{
			return tabela[hashCode].second;
		}
		index = index + 1;
	} while (index < size);
	return false;
	return 0;
}


int Colectie::dim() const {
	/* de adaugat */
	return count;
}


bool Colectie::vida() const {
	/* de adaugat */
	return (0 == count);
}

IteratorColectie Colectie::iterator() const {
	return  IteratorColectie(*this);
}


Colectie::~Colectie() {
	/* de adaugat */
	tabela.~vector();
}


