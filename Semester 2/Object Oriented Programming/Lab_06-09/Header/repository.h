#pragma once

#include <vector>
#include "domain.h"

/// <summary>
///		Clasa pentru stocarea produselor
/// </summary>
class Repository
{
	friend class Bucket;
private:
	std::vector < Product > Repo;

public:
	/// <summary>
	///		Adauga un produs in repo
	/// </summary>
	void AddProduct(Product& ProductToAdd);

	/// <summary>
	///		Modifica un produs existent
	/// </summary>
	/// <param name="IdProductToModify">Id-ul produsului care se modifica</param>
	/// <param name="Name">Numele cu care se modifica - optional</param>
	/// <param name="Type">Tipul cu care se modifica - optional</param>
	/// <param name="Producer">Producatorul cu care se modifica - optional</param>
	/// <param name="Price">Pretul cu care se modifica - optional</param>
	void ModifyProduct(
		int			IdProductToModify,
		std::string NewName = "",
		std::string NewType = "",
		std::string NewProducer = "",
		int			NewPrice = 0
	);

	/// <summary>
	///		Sterge un produs din repo cu id-ul Id
	/// </summary>
	/// <param name="Id">Id-ul produsului care se sterge</param>
	void DeleteProduct(int IdToDelete);
	
	// -------------------------------------------------------------------------------------------

	/// <summary>
	///		Cauta un produs dupa un Id
	/// </summary>
	/// <param name="Id">Id-ul dupa care se cauta</param>
	/// <returns>
	///		Produsul cu respectivul id	- daca exista
	///		Un produs invalid			- altfel
	/// </returns>
	Product FindProductAfterID(int IdToFind);

	/// <summary>
	///		Cauta produse dupa Nume
	/// </summary>
	/// <param name="Name">Numele dupa care se cauta</param>
	/// <returns>vector cu toate produsele care au numele Name</returns>
	std::vector < Product > FindProductsAfterName(std::string NameToFind);

	/// <summary>
	///		Cauta produse dupa Tip
	/// </summary>
	/// <param name="Type">Tipul dupa care se cauta</param>
	/// <returns>vector cu toate produsele cu tipul Type</returns>
	std::vector < Product > FindProductsAfterType(std::string TypeToFind);

	/// <summary>
	///		Cauta produse dupa Producator
	/// </summary>
	/// <param name="Producer">Producatorul dupa care se cauta</param>
	/// <returns>vector cu toate produsele cu tipul Producer</returns>
	std::vector < Product > FindProductsAfterProducer(std::string ProducerToFind);

	/// <summary>
	///		Cauta produse dupa Pret
	/// </summary>
	/// <param name="Price">Pretul dupa care se cauta</param>
	/// <returns>vector cu toate produsele cu pretul Price</returns>
	std::vector < Product > FindProductsAfterPrice(int PriceToFind);

	// -------------------------------------------------------------------------------------------

	/// <summary>
	///		Getter pentru toate produsele
	/// </summary>
	/// <returns>vector cu toate produsele din repo</returns>
	std::vector < Product > GetAll() noexcept;

	/// <summary>
	///		Getter pentru numarul de produse din repo
	/// </summary>
	/// <returns>numar natural - numarul de produse din repo</returns>
	int GetSize() noexcept;
};

class Bucket
{
private:
	std::vector < Product > MemoryBucket;
public:
	void add(Product&);
	void clear();
	void generate(int);
	void exp(std::string);
};