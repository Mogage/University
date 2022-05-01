#include "validator.h"

bool Validator::ValidateString(const std::string& String)
{
	return (String != "");
}

bool Validator::ValidateNumber(int Number) noexcept
{
	return (Number >= 0);
}

void Validator::ValidateProduct(const Product& ProductToVerify)
{
	std::string errors;

	if (false == ValidateNumber(ProductToVerify.GetId()))
	{
		errors = errors + "Id-ul produsului este invalid.\n";
	}
	if (false == ValidateString(ProductToVerify.GetName()))
	{
		errors = errors + "Numele produsului este invalid.\n";
	}
	if (false == ValidateString(ProductToVerify.GetType()))
	{
		errors = errors + "Tipul produsului este invalid.\n";
	}
	if (false == ValidateString(ProductToVerify.GetProducer()))
	{
		errors = errors + "Producatorul produsului este invalid.\n";
	}
	if (false == ValidateNumber(ProductToVerify.GetPrice()))
	{
		errors = errors + "Pretul produsului este invalid.\n";
	}

	if (0 != errors.size())
	{
		throw errors;
	}
}
