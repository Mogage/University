#include "exceptions.h"
#include "repository.h"
#include <algorithm>
#include <fstream>
#include <sstream>

void Repository::loadFromFile()
{
	std::ifstream in(FilePath);
	int id, rank;
	string line, title, artist;

	if (!in.is_open())
		return;

	while (std::getline(in, line))
	{
		std::istringstream stringIn(line);
		stringIn >> id;
		std::getline(stringIn, line, ',');
		std::getline(stringIn, line, ',');
		std::getline(stringIn, artist, ',');
		stringIn >> rank;

		Song toAdd{ id, line, artist, rank };
		Songs.push_back(toAdd);
	}

	in.close();
}

void Repository::writeToFile()
{
	std::ofstream out(FilePath);

	if (!out.is_open())
		return;

	for (auto& song : Songs)
	{
		out << song;
	}

	out.close();
}

Repository::Repository()
{
	FilePath = "saveFiles/out.txt";
	loadFromFile();
}

Repository::Repository(string FileName)
{
	FilePath = "saveFiles/" + FileName;
	loadFromFile();
}

void Repository::update(int Id, int Rank)
{
	for (auto& song : Songs)
	{
		if (song.id() == Id)
		{
			song.setRank(Rank);
			break;
		}
	}
	std::sort(Songs.begin(), Songs.end(), [](const Song& Song1, const Song& Song2) { return Song1.rank() < Song2.rank(); });
	writeToFile();
}

void Repository::update(int Id, string Title)
{
	for (auto& song : Songs)
	{
		if (song.id() == Id)
		{
			song.setTitle(Title);
			break;
		}
	}
	writeToFile();
}

void Repository::deleteId(int Id)
{
	auto it = Songs.begin();
	for (; it != Songs.end(); ++it)
	{
		if ((*it).id() == Id)
			break;
	}
	if (it == Songs.end())
		return;
	int nrSongs = 0;
	for (auto& song : Songs)
	{
		if ((*it).artist() == song.artist())
			nrSongs++;
	}
	if (1 == nrSongs)
		throw RepositoryError("Nu se poate sterge ultima piesa a unui artist.\n");

	Songs.erase(it);
	writeToFile();
}

Song Repository::findId(int Id)
{
	Song toRet;
	for(auto& song : Songs)
	{
		if (Id == song.id())
		{
			toRet = song;
			break;
		}
	}

	return toRet;
}

int Repository::size()
{
	return (int)Songs.size();
}

vector<Song> Repository::all()
{
	return Songs;
}
