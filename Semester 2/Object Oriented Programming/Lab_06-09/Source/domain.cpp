#include "domain.h"
#include <iostream>

Product::Product(
	int		Id, 
	std::string Name,
	std::string Type, 
	std::string Producer, 
	int			Price
	) noexcept 
{
	this->Id = Id;
	this->Name = Name;
	this->Type = Type;
	this->Producer = Producer;
	this->Price = Price;
	// std::cout << "constructor\t";
}

int Product::GetId() const noexcept
{
	return this->Id;
}

std::string Product::GetName() const
{
	return this->Name;
}

std::string Product::GetType() const
{
	return this->Type;
}

std::string Product::GetProducer() const
{
	return this->Producer;
}

int Product::GetPrice() const noexcept
{
	return this->Price;
}

void Product::SetName(std::string NewName)
{
	this->Name = NewName;
}

void Product::SetType(std::string NewType)
{
	this->Type = NewType;
}

void Product::SetProducer(std::string NewProducer)
{
	this->Producer = NewProducer;
}

void Product::SetPrice(int NewPrice) noexcept
{
	this->Price = NewPrice;
}

std::string Product::Print() const
{
	if (this->Id == -1)
	{
		return "Produs invalid.\n";
	}

	return	"Id: " + std::to_string(this->Id) + 
			", Nume: " + this->Name +
			", Tip: " + this->Type + 
			", Producator: " + this->Producer + 
			", Pret: " + std::to_string(this->Price) + ".\n";
}

bool Product::operator!=(const Product& ProductToCompare) noexcept
{
	if (this->Id == ProductToCompare.Id)
	{
		return false;
	}

	if (this->Name == ProductToCompare.Name &&
		this->Type == ProductToCompare.Type &&
		this->Producer == ProductToCompare.Producer)
	{
		return false;
	}

	return true;
}

bool Product::operator==(const Product& ProductToCompare) noexcept
{
	if (this->Id == ProductToCompare.Id)
	{
		return true;
	}

	if (this->Name == ProductToCompare.Name &&
		this->Type == ProductToCompare.Type &&
		this->Producer == ProductToCompare.Producer)
	{
		return true;
	}

	return false;
}

/*
Product& Product::operator=(Product& Given)
{
	this->Id = Given.Id;
	this->Name = Given.Name;
	this->Type = Given.Type;
	this->Producer = Given.Producer;
	this->Price = Given.Price;
	return *this;
}
*/
