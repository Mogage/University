#pragma once

#include "repository.h"
#include "validator.h"

class Service
{
private:
	Repository& Repo;
	Validator& Valid;
public:
	Service(Repository& _Repo, Validator& _Valid);

	void addProduct(int Id, string Name, string Type, double Price);
	vector < Produs > all();
};

