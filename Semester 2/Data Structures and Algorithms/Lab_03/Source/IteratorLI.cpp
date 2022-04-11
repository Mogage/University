#include "IteratorLI.h"
#include "LI.h"
#include <exception>
#include <iostream>

// Teta(1)
IteratorLI::IteratorLI(const LI& li) : lista(li) {
    /* de adaugat */
    this->current = li.First;
}

// Teta(1)
void IteratorLI::prim(){
 	/* de adaugat */
    this->current = this->lista.First;
}

// Teta(1)
void IteratorLI::urmator(){
 	/* de adaugat */
    if (false == this->valid())
    {
        throw std::exception();
    }
    this->current = this->lista.Next[this->current];
}

// Teta(1)
bool IteratorLI::valid() const{
 	/* de adaugat */
    return (this->current != -1);
}

// Teta(1)
TElem IteratorLI::element() const{
 	/* de adaugat */
    if (false == this->valid())
    {
        throw std::exception();
    }
    return this->lista.Elems[this->current];
}
