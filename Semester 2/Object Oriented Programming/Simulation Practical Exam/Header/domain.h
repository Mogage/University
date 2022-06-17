#pragma once

#include <string>

using std::string;

class Telefon
{
private:
	string Cod, Brand, Model;
	int Pret;
public:
	/// <summary>
	///		Constructor default
	/// </summary>
	Telefon();
	/// <summary>
	///		Constructor cu atribute
	/// </summary>
	Telefon(string _CodProdus, string _Brand, string _Model, int _Pret);

	/// <summary>
	///		Getter pentru codul telefonului
	/// </summary>
	/// <returns>string - cod telefon</returns>
	string cod() const;

	/// <summary>
	///		Getter pentru brandul telefonului
	/// </summary>
	/// <returns>string - brand telefon</returns>
	string brand() const;

	/// <summary>
	///		Getter pentru modelul telefonului
	/// </summary>
	/// <returns>string - model telefon</returns>
	string model() const;

	/// <summary>
	///		Getter pentru pretul telefonului
	/// </summary>
	/// <returns>int - pret telefon</returns>
	int pret() const;

	/// <summary>
	///		Seteaza pret
	/// </summary>
	/// <param name="pretNou">noul pret</param>
	void setPret(int pretNou);
};

