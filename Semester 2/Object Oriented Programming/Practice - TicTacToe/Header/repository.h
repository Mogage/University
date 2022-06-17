#pragma once

#include "domain.h"

class Repository
{
private:
	vector < Game > Repo;
	string FilePath;

	void loadFromFile();
	void writeToFile();
public:
	Repository();
	Repository(string _FileName);

	void addGame(Game& GameToAdd);
	Game findId(int Id);
	char move(int Id, int Row, int Col);
	int size();
	vector < Game >& getAll();
};

