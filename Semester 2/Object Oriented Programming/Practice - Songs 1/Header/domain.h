#pragma once

#include <string>
#include <ostream>

using std::string;

class Song
{
private:
	int Id, Rank;
	string Title, Artist;
public:
	Song();
	Song(int _Id, string _Title, string _Artist, int _Rank);

	int id() const;
	int rank() const;
	string title() const;
	string artist() const;

	void setRank(int _Rank);
	void setTitle(string _Title);

	friend std::ostream& operator <<(std::ostream& Os, const Song& ToPrint);
};

