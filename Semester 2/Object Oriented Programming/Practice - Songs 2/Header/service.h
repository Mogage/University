#pragma once

#include "repository.h"

class Service
{
private:
	Repository& Repo;
public:
	Service(Repository& _Repo);

	void add(string Title, string Artist, string Gen);
	void del(int Id);

	vector < Song > all();
};

