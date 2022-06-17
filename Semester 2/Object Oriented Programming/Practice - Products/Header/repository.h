#pragma once

#include "domain.h"
#include <vector>

using std::vector;

class Repository
{
private:
	vector < Produs > Products;
	string FilePath;

	void loadFromFile();
	void writeToFile();
public:
	Repository();
	Repository(string FileName);

	void add(Produs ToAdd);
	int size();
	vector < Produs > all();
};

