#include "IteratorColectie.h"
#include "Colectie.h"


IteratorColectie::IteratorColectie(const Colectie& c): col(c) {
	/* de adaugat */
	curent = { 0, 1 };
	moveToNext();
}

void IteratorColectie::moveToNext()
{
	while (curent.first < col.size && col.tabela[curent.first].second == NULL_TELEM)
	{
		curent.first = curent.first + 1;
	}
}

void IteratorColectie::prim() {
	/* de adaugat */
	curent = { 0, 1 };
	moveToNext();
}


void IteratorColectie::urmator() {
	/* de adaugat */
	if (curent.first >= col.size)
	{
		// throw std::exception();
		return;
	}

	if (curent.second < col.tabela[curent.first].second)
	{
		curent.second = curent.second + 1;
		return;
	}

	curent.second = 1;
	curent.first = curent.first + 1;
	moveToNext();
}


bool IteratorColectie::valid() const {
	/* de adaugat */
	return (curent.first < col.size);
}


TElem IteratorColectie::element() const {
	/* de adaugat */
	if (curent.first >= col.size)
	{
		return -1;
	}

	return col.tabela[curent.first].first;
}
