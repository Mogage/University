#include "repository.h"
#include "exceptions.h"
#include <fstream>
#include <sstream>
#include <algorithm>

void Repository::loadFromFile()
{
	std::ifstream in(FilePath);
	int id;
	double price;
	string line, name, type;

	if (!in.is_open())
		return;

	while (std::getline(in, line))
	{
		std::istringstream stringInput(line);
		stringInput >> id;
		std::getline(stringInput, name, ',');
		std::getline(stringInput, name, ',');
		std::getline(stringInput, type, ',');
		stringInput >> price;

		Produs toAdd{ id, name, type, price };
		Products.push_back(toAdd);
	}

	in.close();
}

void Repository::writeToFile()
{
	std::ofstream out(FilePath);

	if (!out.is_open())
		return;

	for (auto& products : Products)
	{
		out << products;
	}

	out.close();
}

Repository::Repository()
{
	FilePath = "save_files/out.txt";
	loadFromFile();
}

Repository::Repository(string FileName)
{
	FilePath = "save_files/" + FileName;
	loadFromFile();
}

void Repository::add(Produs ToAdd)
{
	for (auto& product : Products)
	{
		if (product == ToAdd)
		{
			throw RepositoryError("Produs existent.\n");
		}
	}

	Products.push_back(ToAdd);
	std::sort(Products.begin(), Products.end(), [](const Produs& Product1, const Produs& Product2) {return Product1.price() < Product2.price(); });
	writeToFile();
}

int Repository::size()
{
	return (int)Products.size();
}

vector<Produs> Repository::all()
{
	return Products;
}
