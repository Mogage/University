#pragma once

#include <vector>
#include <map>
#include "domain.h"

class RepoAbstract
{
public:
	virtual void AddProduct(Product& ProductToAdd) = 0;

	virtual void ModifyProduct(
		int			IdProductToModify,
		std::string NewName = "",
		std::string NewType = "",
		std::string NewProducer = "",
		int			NewPrice = 0
	) = 0;

	virtual void DeleteProduct(int IdToDelete) = 0;

	virtual Product FindProductAfterID(int IdToFind) const = 0;

	virtual std::vector < Product > FindProductsAfterName(std::string NameToFind) const = 0;

	virtual std::vector < Product > FindProductsAfterType(std::string TypeToFind) const = 0;

	virtual std::vector < Product > FindProductsAfterProducer(std::string ProducerToFind) const = 0;

	virtual std::vector < Product > FindProductsAfterPrice(int PriceToFind) const = 0;

	virtual std::vector < Product > GetAll() const = 0;

	virtual int GetSize() const = 0;

	virtual ~RepoAbstract() {}
};

/// <summary>
///		Clasa pentru stocarea produselor
/// </summary>
class Repository : public RepoAbstract
{
	friend class ServiceBucket;
private:
	std::vector < Product > Repo;

public:
	/// <summary>
	///		Adauga un produs in repo
	/// </summary>
	void AddProduct(Product& ProductToAdd) override;

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
	) override;

	/// <summary>
	///		Sterge un produs din repo cu id-ul Id
	/// </summary>
	/// <param name="Id">Id-ul produsului care se sterge</param>
	void DeleteProduct(int IdToDelete) override;
	
	// -------------------------------------------------------------------------------------------

	/// <summary>
	///		Cauta un produs dupa un Id
	/// </summary>
	/// <param name="Id">Id-ul dupa care se cauta</param>
	/// <returns>
	///		Produsul cu respectivul id	- daca exista
	///		Un produs invalid			- altfel
	/// </returns>
	Product FindProductAfterID(int IdToFind) const override;

	/// <summary>
	///		Cauta produse dupa Nume
	/// </summary>
	/// <param name="Name">Numele dupa care se cauta</param>
	/// <returns>vector cu toate produsele care au numele Name</returns>
	std::vector < Product > FindProductsAfterName(std::string NameToFind) const override;

	/// <summary>
	///		Cauta produse dupa Tip
	/// </summary>
	/// <param name="Type">Tipul dupa care se cauta</param>
	/// <returns>vector cu toate produsele cu tipul Type</returns>
	std::vector < Product > FindProductsAfterType(std::string TypeToFind) const override;

	/// <summary>
	///		Cauta produse dupa Producator
	/// </summary>
	/// <param name="Producer">Producatorul dupa care se cauta</param>
	/// <returns>vector cu toate produsele cu tipul Producer</returns>
	std::vector < Product > FindProductsAfterProducer(std::string ProducerToFind) const override;

	/// <summary>
	///		Cauta produse dupa Pret
	/// </summary>
	/// <param name="Price">Pretul dupa care se cauta</param>
	/// <returns>vector cu toate produsele cu pretul Price</returns>
	std::vector < Product > FindProductsAfterPrice(int PriceToFind) const override;

	// -------------------------------------------------------------------------------------------

	/// <summary>
	///		Getter pentru toate produsele
	/// </summary>
	/// <returns>vector cu toate produsele din repo</returns>
	std::vector < Product > GetAll() const override;

	/// <summary>
	///		Getter pentru numarul de produse din repo
	/// </summary>
	/// <returns>numar natural - numarul de produse din repo</returns>
	int GetSize() const override;
};

/// <summary>
///		Stocare produse in fisiere
/// </summary>
class FileRepository : public Repository
{
private:
	std::string FilePath;

	/// <summary>
	///		Incarca din fisier toate produsele
	/// </summary>
	void LoadFromFile();

	/// <summary>
	///		Scrie in fisier toate produsele
	/// </summary>
	void WriteToFile();
public:
	/// <summary>
	///		Constructor pentru un repo de produse in fisiere
	/// </summary>
	/// <param name="FileName">Numele fisierului in care se vor stoca produsele</param>
	FileRepository(std::string FileName);

	/// <summary>
	///		Suprascrie functia de adauga din repo-ul simplu
	/// </summary>
	/// <param name="ProductToAdd">Produsul de adaugat</param>
	void AddProduct(Product& ProductToAdd) override;

	/// <summary>
	///		Suprascrie functia de modifica din repo-ul simplu
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
	) override;

	/// <summary>
	///		Suprascrie functia de stergere din repo-ul simplu
	/// </summary>
	/// <param name="IdToDelete">Id-ul produsului de sters</param>
	void DeleteProduct(int IdToDelete) override;
};

/// <summary>
///		Cos cu produse
/// </summary>
class Bucket 
{
private:
	int TotalPrice;
	std::vector < Product > MemoryBucket;
public:
	/// <summary>
	///		Constructor cos cu produse
	/// </summary>
	Bucket() { TotalPrice = 0; }

	/// <summary>
	///		Adauga un produs in cos
	/// </summary>
	/// <param name="">Produsul de adaugat</param>
	/// <returns>Pretul curent al cosului</returns>
	int add(Product&);

	/// <summary>
	///		Goleste cosul
	/// </summary>
	/// <returns>Returneaza pretul cosului</returns>
	int clear();

	/// <summary>
	///		Returneaza toate produsele din cos
	/// </summary>
	const std::vector < Product >& getBucket() const;

	/// <summary>
	///		Returneaza pretul total al cosului
	/// </summary>
	int getPrice() const;
};

class NewRepository : public RepoAbstract
{
private:
	double Probability;
	std::map < int, Product > Repo;

	void AruncaExceptie() const;
public:
	NewRepository(double _Probability) : Probability{ _Probability } {}

	void AddProduct(Product& ProductToAdd) override;

	void ModifyProduct(
		int			IdProductToModify,
		std::string NewName = "",
		std::string NewType = "",
		std::string NewProducer = "",
		int			NewPrice = 0
	) override;

	void DeleteProduct(int IdToDelete) override;

	Product FindProductAfterID(int IdToFind) const override;

	std::vector < Product > FindProductsAfterName(std::string NameToFind) const override;

	std::vector < Product > FindProductsAfterType(std::string TypeToFind) const override;

	std::vector < Product > FindProductsAfterProducer(std::string ProducerToFind) const override;

	std::vector < Product > FindProductsAfterPrice(int PriceToFind) const override;

	std::vector < Product > GetAll() const override;

	int GetSize() const override;
};

