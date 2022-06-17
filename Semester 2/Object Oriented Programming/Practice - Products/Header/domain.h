#pragma once

#include <string>
#include <ostream>

using std::string;

class Produs
{
private:
	int Id;
	string Name, Type;
	double Price;
public:
	Produs();
	Produs(int _Id, string _Name, string _Type, double _Price);

	int id() const;
	string name();
	string type();
	double price() const;

	bool operator==(const Produs& ToCheck);
	friend std::ostream& operator<<(std::ostream& Os, const Produs& ToPrint);
};

