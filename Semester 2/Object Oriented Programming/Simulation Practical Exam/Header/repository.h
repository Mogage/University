#pragma once

#include "domain.h"
#include <vector>

using std::vector;

class FileRepository
{
private:
	vector < Telefon > Repo;
	string FilePath;

	/// <summary>
	///		Citeste din fisier telefoanele si le adauga in Repo
	/// </summary>
	void loadFromFile();
public:
	/// <summary>
	///		Constructor pentru FileRepo, defineste FilePath-ul
	/// </summary>
	FileRepository();

	/// <summary>
	///		Constructor pentru FileRepo cu numele fisierului
	/// </summary>
	/// <param name="FileName"></param>
	FileRepository(string FileName);

	/// <summary>
	///		Cauta un produs dupa cod
	/// </summary>
	/// <param name="CodToFind">Codul dupa care se cauta</param>
	/// <returns>Telefonul cu codul respectiv, daca exista, un telefon invalid altfel</returns>
	Telefon findAfterCod(const string& CodToFind) const;

	/// <summary>
	///		Getter pentru Telefoanele din Repo
	/// </summary>
	/// <returns>Vector cu toate telefoanele</returns>
	vector < Telefon > getAll() const;

	void addTen(string Brand);
};

