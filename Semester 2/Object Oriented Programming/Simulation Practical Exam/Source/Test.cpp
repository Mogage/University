#include "test.h"

#include <assert.h>

void Test::testDomain()
{
	Telefon empty;
	assert(empty.cod() == "");
	assert(empty.brand() == "");
	assert(empty.model() == "");
	assert(empty.pret() == 0);
	
	Telefon nou{ "1", "Apple", "iPhone", 500 };
	assert(nou.cod() == "1");
	assert(nou.brand() == "Apple");
	assert(nou.model() == "iPhone");
	assert(nou.pret() == 500);

}

void Test::testRepo()
{
	FileRepository repo{ "test.txt" };
	Telefon toFind = repo.findAfterCod("1");
	Telefon empty = repo.findAfterCod("20");
	vector < Telefon > all = repo.getAll();

	assert(toFind.cod() == "1");
	assert(toFind.brand() == "Samsung");
	assert(empty.cod() == "");
	assert(empty.pret() == 0);
	assert(all.size() == 5);
	assert(all[4].cod() == "5");
	assert(all[4].pret() == 5);
}

void Test::testServ()
{
	FileRepository repo{ "test.txt" };
	Service serv{ repo };
	Telefon toFind = serv.findTelefon("1");
	Telefon empty = serv.findTelefon("20");
	vector < Telefon > all = serv.getAll();

	assert(toFind.cod() == "1");
	assert(toFind.brand() == "Samsung");
	assert(empty.cod() == "");
	assert(empty.pret() == 0);
	assert(all.size() == 5);
	assert(all[4].cod() == "5");
	assert(all[4].pret() == 5);


	vector < Telefon > sortedBrand = serv.getSorted([](const Telefon& Tel1, const Telefon& Tel2) { return Tel1.brand() < Tel2.brand(); });
	vector < Telefon > sortedModel = serv.getSorted([](const Telefon& Tel1, const Telefon& Tel2) { return Tel1.model() < Tel2.model(); });
	vector < Telefon > sortedPret = serv.getSorted([](const Telefon& Tel1, const Telefon& Tel2) { return Tel1.pret() < Tel2.pret(); });

	assert(sortedBrand[0].cod() == "4");
	assert(sortedBrand[0].brand() == "Apple");
	assert(sortedBrand[4].cod() == "3");
	assert(sortedBrand[4].brand() == "Samsung");
	assert(sortedModel[0].cod() == "1");
	assert(sortedModel[0].model() == "Galaxy");
	assert(sortedModel[4].cod() == "5");
	assert(sortedModel[4].model() == "eftin");
	assert(sortedPret[0].cod() == "5");
	assert(sortedPret[0].pret() == 5);
	assert(sortedPret[4].cod() == "2");
	assert(sortedPret[4].pret() == 1200);
}

void Test::run()
{
	testDomain();
	testRepo();
	testServ();
}
