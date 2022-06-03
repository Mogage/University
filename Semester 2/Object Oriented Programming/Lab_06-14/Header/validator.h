#pragma once

#include "domain.h"

class Validator
{
public:

	/// <summary>
	///		Valideaza un string sa nu fie null
	/// </summary>
	/// <param name="String">String de verificat</param>
	/// <returns>
	///		true - daca stringul nu e null
	///		false - altfel
	/// </returns>
	bool ValidateString(const std::string& String);

	/// <summary>
	///		Valideaza un numar sa fie pozitiv
	/// </summary>
	/// <param name="Number">Numar de verificat</param>
	/// <returns>
	///		true - daca numarul e pozitiv
	///		false - altfel
	/// </returns>
	bool ValidateNumber(int Number) noexcept;

	/// <summary>
	///		Valideaza un produs
	/// </summary>
	/// <param name="ProductToVerify">Produs de verificat</param>
	/// throw exception cu atributele nevalide
	void ValidateProduct(const Product& ProductToVerify);
};

