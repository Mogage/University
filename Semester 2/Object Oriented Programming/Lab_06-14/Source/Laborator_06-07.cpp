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
#include <fstream>
#include <sstream>

int main()
{
	{
		Test test;
		// Repository repo;
		FileRepository repo("out.csv");
		Bucket bucket;
		Validator valid;
		Service service(repo, valid);
		ServiceBucket serviceBck(repo, bucket, valid);
		UserInterface userInt(service, serviceBck);

		test.RunAll();
		userInt.Run();
	}
	
	_CrtDumpMemoryLeaks();

	return 0;
}