#include "repository.h"
#include <fstream>
#include <algorithm>

Repository::Repository()
{
	FilePath = "save_files/out.txt";
	loadFromFile();
}

Repository::Repository(string _FileName)
{
	FilePath = "save_files/" + _FileName;
	loadFromFile();
}

void Repository::loadFromFile()
{
	std::ifstream in(FilePath);
	char jucator;
	int id, dim;
	string stare, tabla;

	while (in >> id >> dim >> tabla >> jucator)
	{
		std::getline(in, stare);
		stare.erase(stare.begin());
		Repo.push_back(Game{ id, dim, stare, jucator, tabla });
	}

	in.close();
}

void Repository::writeToFile()
{
	std::ofstream out(FilePath);

	if (!out.is_open())
		return;

	for (const auto& game : Repo)
	{
		out << game;
	}

	out.close();
}

void Repository::addGame(Game& GameToAdd)
{
	Repo.insert(Repo.begin(), GameToAdd);
	writeToFile();
}

Game Repository::findId(int Id)
{
	Game empty;

	for (auto& game : Repo)
	{
		if (Id == game.id())
			return game;
	}

	return empty;
}



char Repository::move(int Id, int Row, int Col)
{
	for (auto& game : Repo)
	{
		if (Id == game.id())
		{
			char aux = game.move(Row, Col);
			std::sort(Repo.begin(), Repo.end(), [](const Game& Game1, const Game& Game2) 
				{
					int id1, id2;
					if (Game1.stare() == "Neinceput")
						id1 = 1;
					else if (Game1.stare() == "In derulare")
						id1 = 2;
					else id1 = 3;

					if (Game2.stare() == "Neinceput")
						id2 = 1;
					else if (Game2.stare() == "In derulare")
						id2 = 2;
					else id2 = 3;

					return id1 < id2;
				}
			);
			writeToFile();
			return aux;
		}
	}
	return ' ';
}

int Repository::size()
{
	return (int)Repo.size();
}

vector<Game>& Repository::getAll()
{
	return Repo;
}
