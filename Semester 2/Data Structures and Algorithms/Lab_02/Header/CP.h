#pragma once
#include <vector>
#include <utility>

typedef int TElem;
typedef int TPrioritate;

typedef std::pair<TElem, TPrioritate> Element;

typedef bool (*Relatie)(TPrioritate p1, TPrioritate p2);
class List;
typedef List* PList;


class List {
	friend class CP;
private:
	Element Value;
	PList Next;
public:
	List(Element, PList);
};


class CP {
private:
	/* aici reprezentarea */
	PList Front;
	Relatie Compare;

public:
	//constructorul implicit
	CP(Relatie r);

	//adauga un element in CP
	void adauga(TElem e, TPrioritate p);

	//acceseaza elementul cel mai prioritar in raport cu relatia de ordine
	//arunca exceptie daca CP e vida
	Element element()  const;

	//sterge elementul cel mai prioritar si il returneaza
	//arunca exceptie daca CP e vida
	Element sterge();

	//verifica daca CP e vida;
	bool vida() const;

	// adauga toate elementele din cp in coada curenta, trebuie ca tipul de prioritate sa fie la fel
	void imbinare(CP& cp);

	// destructorul cozii
	~CP();

};
