#include "service.h"
#include "exceptions.h"
#include <algorithm>
#include <random>

void Service::AddProduct(
	int		Id,
	std::string Name,
	std::string Type,
	std::string Producer,
	int			Price
	)
{
	Product productToAdd(Id, Name, Type, Producer, Price);
	this->Valid.ValidateProduct(productToAdd);
	this->Repo.AddProduct(productToAdd);
}

void Service::ModifyProduct(
	int			IdProductToModify,
	std::string Name,
	std::string Type,
	std::string Producer,
	int			Price
	)
{
	
	if (!this->Valid.ValidateNumber(IdProductToModify))
	{
		throw ValidationError("Id invalid.\n");
	}
	if (!this->Valid.ValidateNumber(Price))
	{
		throw ValidationError("Pret invalid.\n");
	}

	this->Repo.ModifyProduct(IdProductToModify, Name, Type, Producer, Price);
}

Product Service::FindProduct(int Id)
{
	if (!this->Valid.ValidateNumber(Id))
	{
		throw ValidationError("Cod de bare invalid.\n");
	}
	return this->Repo.FindProductAfterID(Id);
}
/*
std::vector < Product > Service::FindProduct(int DataToSearch, int FieldToSearch)
{
	std::vector < Product > result;

	switch (FieldToSearch)
	{
	case 1:
		if (!this->Valid.ValidateNumber(DataToSearch))
		{
			throw std::string("Id invalid.\n");
		}
		result = std::vector<Product>(1, this->Repo.FindProductAfterID(DataToSearch));
		break;
	case 5:
		if (!this->Valid.ValidateNumber(DataToSearch))
		{
			throw std::string("Pret invalid.\n");
		}
		result = this->Repo.FindProductsAfterPrice(DataToSearch);
		break;
	default:
		throw std::string("int - FieldToSearch: Invalid value(1/5).\n");
	}

	return result;
}

std::vector < Product > Service::FindProduct(std::string DataToSearch, int FieldToSearch)
{
	std::vector < Product > result;

	switch (FieldToSearch)
	{
	case 2:
		if (!this->Valid.ValidateString(DataToSearch))
		{
			throw std::string("Nume invalid.\n");
		}
		result = this->Repo.FindProductsAfterName(DataToSearch);
		break;
	case 3:
		if (!this->Valid.ValidateString(DataToSearch))
		{
			throw std::string("Tip invalid.\n");
		}
		result = this->Repo.FindProductsAfterType(DataToSearch);
		break;
	case 4:
		if (!this->Valid.ValidateString(DataToSearch))
		{
			throw std::string("Producator invalid.\n");
		}
		result = this->Repo.FindProductsAfterProducer(DataToSearch);
		break;
	default:
		throw std::string("string - FieldToSearch: Invalid value(2-4).\n");
	}

	return result;
}
*/

void Service::DeleteProduct(int Id)
{
	if (!this->Valid.ValidateNumber(Id))
	{
		throw ValidationError("Id invalid.\n");
	}
	this->Repo.DeleteProduct(Id);
}

std::vector<Product> Service::Filter(int Price) const
{
	if (!this->Valid.ValidateNumber(Price))
	{
		throw ValidationError("Pret invalid.\n");
	}

	return this->Repo.FindProductsAfterPrice(Price);
}

std::vector<Product> Service::Filter(std::string StringData, int FilterType)
{
	if (!this->Valid.ValidateString(StringData))
	{
		throw ValidationError("Date invalide.\n");
	}
	if (FilterType < 1 || FilterType > 2)
	{
		throw ValidationError("Tip filtru invalid.\n");
	}
	
	if (FilterType == 1)
	{
		return this->Repo.FindProductsAfterName(StringData);
	}
	
	return this->Repo.FindProductsAfterProducer(StringData);
}

std::vector<Product> Service::Sort(bool Compare(const Product& Product1, const Product& Product2))
{
	std::vector < Product > products = this->Repo.GetAll();

	std::sort(products.begin(), products.end(), Compare);
	
	return products;
}

std::vector < Product > Service::GetAll() const noexcept
{
	return this->Repo.GetAll();
}

int ServiceBucket::addToBucket(std::string Name)
{
	if (false == Valid.ValidateString(Name))
	{
		throw ValidationError("Nume invalid.\n");
	}
	std::vector < Product > products = Repo.FindProductsAfterName(Name);
	if (products.size() == 0)
	{
		throw RepositoryError("Nu exista niciun produs cu acest nume.\n");
	}
	return Bck.add(products[0]);
}

int ServiceBucket::clearBucket()
{
	return Bck.clear();
}

int ServiceBucket::generateBucket(int NumberOfProducts)
{
	std::mt19937 mt{ std::random_device{}() };
	std::uniform_int_distribution<> dist(0, (int)Repo.GetSize() - 1);
	int rndNr, totalPrice = Bck.getPrice();
	Product random;

	for (int count = 0; count < NumberOfProducts; count = count + 1)
	{
		rndNr = dist(mt);
		random = Repo.Repo[rndNr];
		totalPrice = Bck.add(random);
	}
	return totalPrice;
}

const std::vector < Product >& ServiceBucket::getBucket() const
{
	return Bck.getBucket();
}