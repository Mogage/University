#pragma once

#include "service.h"

class UserInterface
{
private:
	Service& Serv;

	/// <summary>
	///		Afiseaza meniul
	/// </summary>
	void PrintMenu();

	/// <summary>
	///		Citeste si adauga un produs
	/// </summary>
	void Add();

	/// <summary>
	///		Modifica un produs
	/// </summary>
	void Modify();

	/// <summary>
	///		Citeste id-ul si sterge produs cu acel id
	/// </summary>
	void Delete();

	/// <summary>
	///		Afiseaza toate produsele
	/// </summary>
	void PrintAll();

	/// <summary>
	///		Cauta un produs dupa id
	/// </summary>
	void Find();

	/// <summary>
	///		Afiseaza produsele filtrate
	/// </summary>
	void Filter();

	/// <summary>
	///		Afiseaza produsele sortate
	/// </summary>
	void Sort();

public:
	/// <summary>
	///		Constructor user interface
	/// </summary>
	/// <param name="Serv">Service-ul de produse</param>
	UserInterface(Service& Serv) noexcept : Serv{ Serv } {}

	/// <summary>
	///		Main app start
	/// </summary>
	void Run();
};

