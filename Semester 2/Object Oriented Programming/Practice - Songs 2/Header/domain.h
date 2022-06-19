#pragma once

#include <string>

using std::string;

class Song
{
private:
	int Id;
	string Title, Artist, Gen;
public:
	Song();
	Song(int _Id, string _Title, string _Artist, string _Gen);

	int id() const;
	string title() const;
	string artist() const;
	string gen() const;

	friend std::ostream& operator << (std::ostream& Os, const Song& _Song);
};

