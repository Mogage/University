/*
Creati o aplicatie care permite gestiunea ofertelor de la o agentie de turism.

Fiecare oferta are: tip (munte, mare, citiy break), destinatie, data plecare, pret

Aplicatia permite:

 a) Adaugarea de noi oferte.

 b) Actualizare oferte

 c) Stergere oferta

 d) Vizualizare oferete ordonat dupa pret, destinatie (crescator/descrescator)

 e) Vizualizare oferta filtrate dupa un criteriu (destinatie, tip, pret)
 */

#include "common.h"
#include "userInterface.h"
#include "tests.h"
#include <crtdbg.h>

#define _CRTDBG_MAP_ALLOC 

int main()
{
	RunAllTests();

	REPOSITORY OfferRepo = NULL;
	SERVICE OfferService = NULL;
	USERINT OfferUserInt = NULL;

	CreateRepo(&OfferRepo);
	CreateService(&OfferService, OfferRepo);
	CreateUserInt(&OfferUserInt, OfferService);

	RunApp(OfferUserInt);

	DestroyRepo(&OfferRepo);
	DestroyService(&OfferService);
	DestroyUserInt(&OfferUserInt);

	_CrtDumpMemoryLeaks();

	return 0;
}