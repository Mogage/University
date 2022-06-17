#include "service.h"

#include <algorithm>

Telefon Service::findTelefon(const string& CodToFind) const
{
    return Repo.findAfterCod(CodToFind);
}

vector<Telefon> Service::getSorted(bool Compare(const Telefon& Product1, const Telefon& Product2))
{
	std::vector < Telefon > products = this->Repo.getAll();

	std::sort(products.begin(), products.end(), Compare);

	return products;
}

vector<Telefon> Service::getAll()
{
    return Repo.getAll();
}

void Service::addTen(string Brand)
{
	Repo.addTen(Brand);
}
