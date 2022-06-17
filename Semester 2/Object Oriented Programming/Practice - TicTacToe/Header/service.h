#pragma once

#include "repository.h"
#include "validator.h" 

class Service
{
private:
	Repository& Repo;
	Validator& Valid;
public:
	Service( Repository & _Repo, Validator & _Valid ) : Repo{ _Repo }, Valid{ _Valid } {}

	void createGame(int Dim, string TablaSir, char Jucator);
	char move(int Id, int Row, int Col);
	Game find(int Id);
	vector < Game >& all();
};

