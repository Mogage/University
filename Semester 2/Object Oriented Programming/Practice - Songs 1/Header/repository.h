#pragma once

#include "domain.h"
#include <vector>

using std::vector;

class Repository
{
private:
	vector < Song > Songs;
	string FilePath;

	void loadFromFile();
	void writeToFile();
public:
	Repository();
	Repository(string FileName);

	void update(int Id, int Rank);
	void update(int Id, string Title);
	void deleteId(int Id);
	Song findId(int Id);

	int size();
	vector < Song > all();
};

