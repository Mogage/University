/*
Creați o aplicație care permite:
· gestiunea unei liste de produse. Produs: nume, tip, preț, producător
· adăugare, ștergere, modificare și afișare produse
· căutare produs
· filtrare produse după: preț, nume, producător
· sortare produse după: nume, preț, nume + tip
*/

#include <iostream>
#include "userinterface.h"
#include "test.h"
#include <crtdbg.h>

int main()
{
	Test test;
	Repository repo;
	Validator valid;
	Service service(repo, valid);
	UserInterface userInt(service);

	test.RunAll();
	userInt.Run();

	_CrtDumpMemoryLeaks();

	return 0;
}