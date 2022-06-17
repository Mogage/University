#include "service.h"

Service::Service(Repository& _Repo, Validator& _Valid) :
	Repo{_Repo}, Valid{_Valid}
{}

void Service::addProduct(int Id, string Name, string Type, double Price)
{
	Produs product{ Id, Name, Type, Price };
	Valid.ValidateProduct(product);
	Repo.add(product);
}

vector<Produs> Service::all()
{
	return Repo.all();
}
