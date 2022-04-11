#include "IteratorColectie.h"
#include "Colectie.h"
#include <exception>

/*  Complexitate Constructor: Teta(1)*/
IteratorColectie::IteratorColectie(const Colectie& c): col(c) {
	/* de adaugat */
	this->current.currentIndex = 0;
	this->current.currentFreq = 0;
}

/*  Complexitate prim: Teta(1)*/
void IteratorColectie::prim() {
	/* de adaugat */
	this->current.currentIndex = 0;
	this->current.currentFreq = 0;
}

/*  Complexitate urmator: Teta(1)*/
void IteratorColectie::urmator() {
	/* de adaugat */
	if (false == this->valid())
	{
		throw std::exception("Elementul nu este valid");
	}

	if (this->current.currentFreq < this->col.Array[this->current.currentIndex].Frequency - 1)
	{
		this->current.currentFreq = this->current.currentFreq + 1;
	}
	else 
	{
		this->current.currentIndex = this->current.currentIndex + 1;
		this->current.currentFreq = 0;
	}
}

/*  Complexitate valid: Teta(1)*/
bool IteratorColectie::valid() const {
	/* de adaugat */
	return (this->current.currentIndex < this->col.Count);
}

/*  Complexitate element: Teta(1)*/
TElem IteratorColectie::element() const {
	/* de adaugat */
	if (false == this->valid())
	{
		throw std::exception("Elementul nu este valid");
	}
	return this->col.Array[this->current.currentIndex].Value;
}
