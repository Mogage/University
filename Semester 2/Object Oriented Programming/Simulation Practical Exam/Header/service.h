#pragma once

#include "repository.h"

class Service
{
private:
	FileRepository& Repo;
public:
	/// <summary>
	///		Constructor pentru service
	/// </summary>
	/// <param name="_Repo">Repo cu care se lucreaza</param>
	Service(FileRepository& _Repo) : Repo{ _Repo } {}

	/// <summary>
	///		Cauta un telefon in Repo
	/// </summary>
	/// <param name="CodToFind">Codul dupa care se cauta</param>
	/// <returns>Telefonul cu codul respectiv, daca exista, un telefon invalid altfel</returns>
	Telefon findTelefon(const string& CodToFind) const;

	/// <summary>
	///		Sorteaza telefoanele dupa un criteriu
	/// </summary>
	/// <param name="Compare">Functia de comparare</param>
	/// <returns>Telefoanele sortare conform functiei de comparare</returns>
	vector < Telefon > getSorted(bool Compare(const Telefon& Product1, const Telefon& Product2));

	/// <summary>
	///		Getter pentru toate telefoanele din repo
	/// </summary>
	/// <returns>Vector cu toate telefoanele</returns>
	vector < Telefon > getAll();

	void addTen(string Brand);
};

