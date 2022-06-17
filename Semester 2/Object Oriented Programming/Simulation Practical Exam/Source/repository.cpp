#include "repository.h"

#include <fstream>
#include <sstream>

void FileRepository::loadFromFile()
{
	std::ifstream in(FilePath);
	string line, copyLine, cod, brand, model;
	int pret;

	while (std::getline(in, line))
	{
		copyLine = line;
		copyLine.erase(std::remove(copyLine.begin(), copyLine.end(), ','), copyLine.end());
		if (copyLine.empty())
			continue;
		std::istringstream stringInput(line);
		std::getline(stringInput, cod, ',');
		std::getline(stringInput, brand, ',');
		std::getline(stringInput, model, ',');
		stringInput >> pret;

		Telefon toAdd{ cod, brand, model, pret };
		Repo.push_back(toAdd);
	}

	in.close();
}

FileRepository::FileRepository()
{
	FilePath = "saveFiles/out.txt";
	loadFromFile();
}

FileRepository::FileRepository(string FileName)
{
	FilePath = "saveFiles/" + FileName;
	loadFromFile();
}

vector<Telefon> FileRepository::getAll() const
{
	return vector<Telefon>(Repo);
}

void FileRepository::addTen(string Brand)
{
	for (auto& telefon : Repo)
	{
		if (Brand == telefon.brand())
		{
			telefon.setPret(telefon.pret() + 10);
		}
	}
}

Telefon FileRepository::findAfterCod(const string& CodToFind) const
{
	Telefon toReturn;

	for (const auto& telefon : Repo)
	{
		if (telefon.cod() == CodToFind)
		{
			toReturn = telefon;
			break;
		}
	}

	return toReturn;
}
