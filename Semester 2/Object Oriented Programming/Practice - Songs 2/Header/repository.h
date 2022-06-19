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
	Repository(const string& FileName);

	void add(const Song& ToAdd);
	void deleteId(int Id);

	int size() const;
	const vector < Song >& all() const;
};

