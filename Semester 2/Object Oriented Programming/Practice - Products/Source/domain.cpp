#include "domain.h"

Produs::Produs()
{
	Id = -1;
	Name = "";
	Type = "";
	Price = -1;
}

Produs::Produs(int _Id, string _Name, string _Type, double _Price) :
	Id{ _Id }, Name{ _Name }, Type{ _Type }, Price{ _Price } 
{
}

int Produs::id() const
{
	return Id;
}

string Produs::name()
{
	return Name;
}

string Produs::type()
{
	return Type;
}

double Produs::price() const
{
	return Price;
}

bool Produs::operator==(const Produs& ToCheck)
{
	return this->Id == ToCheck.Id;
}

std::ostream& operator<<(std::ostream& Os, const Produs& ToPrint)
{
	Os << ToPrint.Id << ',' << ToPrint.Name << ',' << ToPrint.Type << ',' << ToPrint.Price << '\n';
	return Os;
}
