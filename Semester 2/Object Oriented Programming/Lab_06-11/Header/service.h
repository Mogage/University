#pragma once

#include "repository.h"
#include "validator.h"
#include "domain.h"
#include "undo.h"
#include <vector>

/// <summary>
///		Service pentru produse
/// </summary>
class Service
{
private:
	Repository& Repo;
	Validator& Valid;
	std::vector < Undo* > UndoActions;

public:
	/// <summary>
	///		Constructor service
	/// </summary>
	/// <param name="Repo">Repo pentru obiecte</param>
	/// <param name="Valid">Validator pentru obiecte</param>
	Service(Repository& Repo, Validator& Valid) noexcept : Repo{ Repo }, Valid{ Valid } {}

	/// <summary>
	///		Destructor service
	/// </summary>
	~Service();

	/// <summary>
	///		Adauga un produs cu atribute
	/// </summary>
	/// <param name="Id">Id-ul noului produs adaugat</param>
	/// <param name="Name">Numele noului produs adaugat</param>
	/// <param name="Type">Tipul noului produs adaugat</param>
	/// <param name="Producer">Producatorul noului produs adaugat</param>
	/// <param name="Price">Pretul noului produs adaugat</param>
	void AddProduct(
		int			Id,
		std::string Name,
		std::string Type,
		std::string Producer,
		int			Price
	);

	/// <summary>
	///		Modifica un produs cu id-ul dat
	/// </summary>
	/// <param name="IdProductToModify">Id-ul produsului de modificat</param>
	/// <param name="Name">Noul nume daca este diferit de ""</param>
	/// <param name="Type">Noul tip daca este diferit de ""</param>
	/// <param name="Producer">Noul producator daca este diferit de ""</param>
	/// <param name="Price">Noul pret daca este diferit de ""</param>
	void ModifyProduct(
		int			IdProductToModify,
		std::string Name = "",
		std::string Type = "",
		std::string Producer = "",
		int			Price = 0
	);

	/// <summary>
	///		Cauta un produs dupa un Id
	/// </summary>
	/// <param name="Id">Id-ul produsului de cautat</param>
	/// <returns>
	///		Un produs gol daca nu exista niciun produs cu acest id
	///		daca exista produsul respectiv
	/// </returns>
	Product FindProduct(int Id);
	// std::vector < Product > FindProduct(int DataToSearch, int FieldToSearch);
	// std::vector < Product > FindProduct(std::string DataToSearch, int FieldToSearch);

	/// <summary>
	///		Sterge un produs dupa id
	/// </summary>
	/// <param name="Id">Id-ul produsului care se sterge</param>
	void DeleteProduct(int Id);

	/// <summary>
	///		Filtreaza produsele dupa pret
	/// </summary>
	/// <param name="Price">Pretul produselor filtrate</param>
	/// <returns>vector cu produsele cu acest pret</returns>
	std::vector < Product > Filter(int Price) const;

	/// <summary>
	///		Filtreaza produsele dupa nume daca filtertype = 1, dupa producator altfel
	/// </summary>
	/// <param name="StringData">Stringul dupa care se filtreaza</param>
	/// <param name="FilterType">
	///		1 - se filtreaza dupa nume
	///		2 - se filtreaza dupa producator
	///		altceva - throw std::string
	/// </param>
	/// <returns>vector cu produsele filtrate in functie de FilterType</returns>
	std::vector < Product > Filter(std::string StringData, int FilterType);

	/// <summary>
	///		Sorteaza produsele dupa o functie
	/// </summary>
	/// <param name="Compare">Functie de comparare dupa care se sorteaza</param>
	/// <returns>vector cu produsele sortate</returns>
	std::vector < Product > Sort(bool Compare(const Product& Product1, const Product& Product2));

	/// <summary>
	///		Getter pentru toate produsele existente
	/// </summary>
	/// <returns>vector cu toate produsele</returns>
	std::vector < Product > GetAll() const noexcept;

	/// <summary>
	///		Functie pentru a face undo la o operatie
	/// </summary>
	void UndoServ();
};

/// <summary>
///		Service pentru cosul de cumparaturi
/// </summary>
class ServiceBucket
{
private:
	Repository& Repo;
	Bucket& Bck;
	Validator& Valid;
public:
	/// <summary>
	///		Constructor service cos de cumparaturi
	/// </summary>
	/// <param name="_Repo"></param>
	/// <param name="_Bucket"></param>
	/// <param name="_Validator"></param>
	ServiceBucket(Repository& _Repo, Bucket& _Bucket, Validator& _Validator) : Repo{ _Repo }, Bck{ _Bucket }, Valid{ _Validator }{}

	/// <summary>
	///		Adauga in cos un produs dupa nume
	/// </summary>
	/// <param name="">Numele produsului de adaugat</param>
	/// <returns>Pretul curent al cosului</returns>
	int addToBucket(std::string);

	/// <summary>
	///		Goleste cosul
	/// </summary>
	/// <returns>Pretul curent al cosului</returns>
	int clearBucket();

	/// <summary>
	///		Adauga un numbar de produse random in cos
	/// </summary>
	/// <param name="">Numarul de produse random de adaugat</param>
	/// <returns>Valoarea cosului</returns>
	int generateBucket(int);

	/// <summary>
	///		Returneaza toate produsele din cos
	/// </summary>
	const std::vector < Product >& getBucket() const;
};