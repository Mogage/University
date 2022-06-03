#include "IteratorMDO.h"
#include "MDO.h"
#include <iostream>
#include <vector>

#include <exception>
using namespace std;

// Complexitate: Teta(n), n - numarul de elemente initial
MDO::MDO(Relatie r) {
	/* de adaugat */
	Size = INITIAL_SIZE;
	Count = 0;
	Rel = r;
	//Tablou = std::vector < TElem >(Size, { -1, 0 });
	Tablou = new TElem[Size];
	for (int count = 0; count < Size; count = count + 1)
	{
		Tablou[count] = { -1,0 };
	}
}

// Complexitate: O(logn) amortizat
void MDO::adauga(TCheie c, TValoare v) {
	/* de adaugat */
	TElem toAdd = { c,v };
	int currentPosition = 1;
	Count = Count + 1;
	while (currentPosition < Size)
	{
		if (Tablou[currentPosition].first == -1)
		{
			break;
		}
		else
		{
			if (!Rel(Tablou[currentPosition].first, c))
			{
				currentPosition = 2 * currentPosition;
			}
			else if (Rel(Tablou[currentPosition].first, c))
			{
				currentPosition = 2 * currentPosition + 1;
			}
		}
	}
	Tablou[currentPosition] = toAdd;
}

// Complexitate: O(logn) amortizat
vector<TValoare> MDO::cauta(TCheie c) const {
	/* de adaugat */
	vector<TValoare> toReturn;
	int currentPosition = 1;
	while (currentPosition < Size)
	{
		if (Tablou[currentPosition].first == -1)
		{
			break;
		}
		else if (Tablou[currentPosition].first == c)
		{
			toReturn.push_back(Tablou[currentPosition].second);
		}
		if (!Rel(Tablou[currentPosition].first, c))
		{
			currentPosition = 2 * currentPosition;
		}
		else if (Rel(Tablou[currentPosition].first, c))
		{
			currentPosition = 2 * currentPosition + 1;
		}
	}
	return toReturn;
}

// Complexitate: O(logn)
int MDO::FindMin(int Index) const
{
	while (Tablou[2 * Index].first != -1)
	{
		Index = 2 * Index;
	}
	return Index;
}

// Complexitate: O(logn) amortizat
bool MDO::sterge(TCheie c, TValoare v) {
	/* de adaugat */
	
	int currentPosition = 1, min;
	while (currentPosition < Size)
	{
		if (Tablou[currentPosition].first == -1)
		{
			break;
		}
		else if (Tablou[currentPosition].first != c || Tablou[currentPosition].second != v)
		{
			if (!Rel(Tablou[currentPosition].first, c))
			{
				currentPosition = 2 * currentPosition;
			}
			else if (Rel(Tablou[currentPosition].first, c))
			{
				currentPosition = 2 * currentPosition + 1;
			}
		}
		else
		{
			if (Tablou[2 * currentPosition].first != -1 && Tablou[2 * currentPosition + 1].first != -1)
			{
				min = FindMin(2 * currentPosition + 1);
				Tablou[currentPosition] = Tablou[min];
				Tablou[min] = { -1,0 };
			}
			else
			{
				if (Tablou[2 * currentPosition].first == -1)
				{
					while (Tablou[2 * currentPosition + 1].first != -1)
					{
						Tablou[currentPosition] = Tablou[2 * currentPosition + 1];
						currentPosition = 2 * currentPosition + 1;
					}
				}
				else if (Tablou[2 * currentPosition + 1].first == -1)
				{
					while (Tablou[2 * currentPosition].first != -1)
					{
						Tablou[currentPosition] = Tablou[2 * currentPosition];
						currentPosition = 2 * currentPosition;
					}
				}
				Tablou[currentPosition] = { -1,0 };
			}
			Count = Count - 1;
			return true;
		}
	}
	
	return false;
}

// Complexitate: Teta(1)
int MDO::dim() const {
	/* de adaugat */
	return Count;
}

// Complexitate: Teta(1)
bool MDO::vid() const {
	/* de adaugat */
	return (0 == Count);
}

// Complexitate: O(logn)
IteratorMDO MDO::iterator() const {
	return IteratorMDO(*this);
}

// Complexitate: Teta(n)
vector<TValoare> MDO::colectiaValorilor() const
{
	vector<TValoare> toReturn;
	for (int index = 1; index < Size; index = index + 1)
	{
		if (Tablou[index].first != -1)
		{
			toReturn.push_back(Tablou[index].second);
		}
	}
	return toReturn;
}

// Complexitate: Teta(1)
MDO::~MDO() {
	/* de adaugat */
	delete[] Tablou;
}
