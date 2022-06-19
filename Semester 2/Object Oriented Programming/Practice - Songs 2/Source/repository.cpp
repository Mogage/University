#include "repository.h"
#include <fstream>
#include <sstream>

void Repository::loadFromFile()
{
	std::ifstream in(FilePath);
	int id;
	string line, title, artist, gen;

	if (!in.is_open())
		return;

	while (std::getline(in, line))
	{
		std::istringstream lineS(line);
		lineS >> id;
		std::getline(lineS, title, ',');
		std::getline(lineS, title, ',');
		std::getline(lineS, artist, ',');
		std::getline(lineS, gen, ',');

		Songs.push_back(Song{ id, title, artist, gen });
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

Repository::Repository(const string& FileName)
{
	FilePath = "saveFiles/" + FileName;
	loadFromFile();
}

void Repository::add(const Song& ToAdd)
{
	Songs.push_back(ToAdd);
	writeToFile();
}

void Repository::deleteId(int Id)
{
	for (auto it = Songs.begin(); it != Songs.end(); ++it)
	{
		if ((*it).id() == Id)
		{
			Songs.erase(it);
			break;
		}
	}
	writeToFile();
}

int Repository::size() const
{
	return (int)Songs.size();
}

const vector<Song>& Repository::all() const
{
	return Songs;
}
