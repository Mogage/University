#pragma once

#include "repository.h"

class Service
{
private:
	Repository& Repo;
public:
	Service(Repository& _Repo);

	void update(int Id, int Rank);
	void update(int Id, string Title);
	void deleteId(int Id);

	Song find(int Id);

	vector < Song > all();
};

