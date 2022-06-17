#include "domain.h"

Telefon::Telefon()
{
	Cod = "";
	Brand = "";
	Model = "";
	Pret = 0;
}

Telefon::Telefon(string _CodProdus, string _Brand, string _Model, int _Pret) :
	Cod{ _CodProdus }, Brand{ _Brand }, Model{ _Model }, Pret{ _Pret }
{}

string Telefon::brand() const
{
	return Brand;
}

string Telefon::model() const
{
	return Model;
}

int Telefon::pret() const
{
	return Pret;
}

void Telefon::setPret(int pretNou)
{
	Pret = pretNou;
}

string Telefon::cod() const
{
	return Cod;
}
