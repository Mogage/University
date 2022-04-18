#pragma once

#include "service.h"

class UserInterface
{
private:
	Service& Serv;
	ServiceBucket& ServBck;

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

	/// <summary>
	///		Citeste numele produsului de adaugat in cos
	/// </summary>
	void AddBucket();

	/// <summary>
	///		Goleste cosul
	/// </summary>
	void ClearBucket();

	/// <summary>
	///		Citeste un numar x si adauga x produse random in cos
	/// </summary>
	void GenerateBucket();

	/// <summary>
	///		Exporteaza cosul intr-un fisier citit de la utilizator
	/// </summary>
	void ExportBucket();

	/// <summary>
	///		Intra in modul batch de inserare comenzi
	/// </summary>
	void BatchMode();

	/// <summary>
	///		Anuleaza ultima operatie daca este posibil
	/// </summary>
	void Undo();

public:
	/// <summary>
	///		Constructor user interface
	/// </summary>
	/// <param name="Serv">Service-ul de produse</param>
	UserInterface(Service& Serv, ServiceBucket& ServBck) noexcept : Serv{ Serv }, ServBck{ ServBck } {}

	/// <summary>
	///		Main app start
	/// </summary>
	void Run();
};

