#pragma once

#include <string>
#include <vector>

/// <summary>
///		Clasa pentru tipul produs
/// </summary>
class Product
{
private:
	std::string Name, Type, Producer;
	int Price;
	int Id;

public:
	
	/// <summary>
	///		Constructor pentru un tip produs
	/// </summary>
	Product(
		int			Id = -1, 
		std::string Name = "",
		std::string Type = "",
		std::string Producer = "",
		int			Price = -1
	) noexcept;

	// -------------------------------------------------------------------------------------------

	/// <summary>
	///		Getter pentru id-ul unui produs
	/// </summary>
	/// <returns>int - Id-ul produsului respectiv</returns>
	int GetId() const noexcept;

	/// <summary>
	///		Getter pentru numele unui produs
	/// </summary>
	/// <returns>std::string - Numele produsului respectiv</returns>
	std::string GetName() const;

	/// <summary>
	///		Getter pentru tipul unui produs
	/// </summary>
	/// <returns>std::string - Tipul produsului respectiv</returns>
	std::string GetType() const;

	/// <summary>
	///		Getter pentru producatorul unui produs
	/// </summary>
	/// <returns>std::string - Producatorul produsului respectiv</returns>
	std::string GetProducer() const;

	/// <summary>
	///		Getter pentru pretul unui produs
	/// </summary>
	/// <returns>std::string - Pretul produsului respectiv</returns>
	int GetPrice() const noexcept;

	// -------------------------------------------------------------------------------------------

	/// <summary>
	///		Setter pentru numele unui produs
	/// </summary>
	/// <param name="NewName">Noul nume pentru produsul respectiv</param>
	void SetName(std::string NewName);

	/// <summary>
	///		Setter pentru tipul unui produs
	/// </summary>
	/// <param name="NewType">Noul tip pentru produsul respectiv</param>
	void SetType(std::string NewType);

	/// <summary>
	///		Setter pentru producatorul unui produs
	/// </summary>
	/// <param name="NewProducer">Noul producator pentru produsul respectiv</param>
	void SetProducer(std::string NewProducer);

	/// <summary>
	///		Setter pentru pretul unui produs
	/// </summary>
	/// <param name="NewPrice">Noul pret pentru produsul respectiv</param>
	void SetPrice(int NewPrice) noexcept;
	
	// -------------------------------------------------------------------------------------------

	/// <summary>
	///		Formatarea unui produs pentru tiparire
	/// </summary>
	/// <returns>std:string - Formatul de afisare a unui produs</returns>
	std::string Print() const;

	// -------------------------------------------------------------------------------------------

	/// <summary>
	///		Suprascrie operatorul not equal pentru tipul produs
	/// </summary>
	/// <param name="ProductToCompare">Produsul cu care se compara</param>
	/// <returns>
	///		true - daca cele 2 produse sunt diferite
	///		false - altfel
	/// </returns>
	bool operator!=(const Product& ProductToCompare) noexcept;

	/// <summary>
	///		Suprascrie operatorul equal pentru tipul produs
	/// </summary>
	/// <param name="ProductToCompare">Produsul cu care se compara</param>
	/// <returns>
	///		true - daca cele 2 produse sunt egale
	///		false - altfel
	/// </returns>
	bool operator==(const Product& ProductToCompare) noexcept;

	// Product& operator=(Product& Given);
};
