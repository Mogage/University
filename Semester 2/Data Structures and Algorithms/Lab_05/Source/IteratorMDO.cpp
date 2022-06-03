#include "IteratorMDO.h"
#include "MDO.h"

// Complexitate: O(logn)
IteratorMDO::IteratorMDO(const MDO& d) : dict(d){
	/* de adaugat */
	CurrentIndex = dict.FindMin(1);
}

// Complexitate: O(logn)
void IteratorMDO::prim(){
	/* de adaugat */
	CurrentIndex = dict.FindMin(1);
}

// Complexitate: O(logn)
void IteratorMDO::urmator(){
	/* de adaugat */
	if (dict.Tablou[2 * CurrentIndex + 1].first != -1)
	{
		CurrentIndex = dict.FindMin(2 * CurrentIndex + 1);
		return;
	}
	int aux = CurrentIndex, prec = aux / 2;
	while (prec != 0 && aux == 2 * prec + 1)
	{
		aux = prec;
		prec = aux / 2;
	}
	CurrentIndex = prec;
}

// Complexitate: Teta(1)
bool IteratorMDO::valid() const{
	/* de adaugat */
	return (dict.Tablou[CurrentIndex].first != -1);
}

// Complexitate: Teta(1)
TElem IteratorMDO::element() const{
	/* de adaugat */
	return dict.Tablou[CurrentIndex];
}


