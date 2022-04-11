#include "repository.h"
#include <random>

void Repository::AddProduct(Product& ProductToAdd)
{
	for(auto& iterator : Repo)
	{
		if (iterator == ProductToAdd)
		{
			throw std::string("Acest produs exista deja.\n");
		}
	}

	this->Repo.push_back(ProductToAdd);
}

void Repository::ModifyProduct(
	int		IdProductToModify, 
	std::string NewName,
	std::string NewType,
	std::string NewProducer,
	int			NewPrice
	)
{
	bool exist = false;

	for(auto& iterator : Repo)
	{
		if (iterator.GetId() == IdProductToModify)
		{
			if ("" != NewName)
			{
				iterator.SetName(NewName);
			}
			if ("" != NewType)
			{
				iterator.SetType(NewType);
			}
			if ("" != NewProducer)
			{
				iterator.SetProducer(NewProducer);
			}
			if (0 != NewPrice)
			{
				iterator.SetPrice(NewPrice);
			}
			exist = true;
			break;
		}
	}

	if (!exist)
	{
		throw std::string("Nu exista niciun produs cu acest id.\n");
	}
}

Product Repository::FindProductAfterID(int IdToFind)
{
	Product toReturn;
	for(auto& iterator : Repo)
	{
		if (iterator.GetId() == IdToFind)
		{
			toReturn = iterator;
			break;
		}
	}
	return toReturn;
}


std::vector < Product > Repository::FindProductsAfterName(std::string NameToFind)
{
	std::vector < Product > toReturn;
	for(auto& iterator : Repo)
	{
		if (iterator.GetName() == NameToFind)
		{
			toReturn.push_back(iterator);
		}
	}
	return toReturn;
}

std::vector < Product > Repository::FindProductsAfterType(std::string TypeToFind)
{
	std::vector < Product > toReturn;
	for(auto& iterator : Repo)
	{
		if (iterator.GetType() == TypeToFind)
		{
			toReturn.push_back(iterator);
		}
	}
	return toReturn;
}

std::vector < Product > Repository::FindProductsAfterProducer(std::string ProducerToFind)
{
	std::vector < Product > toReturn;
	for(auto& iterator : Repo)
	{
		if (iterator.GetProducer() == ProducerToFind)
		{
			toReturn.push_back(iterator);
		}
	}
	return toReturn;
}

std::vector < Product > Repository::FindProductsAfterPrice(int PriceToFind)
{
	std::vector < Product > toReturn;
	for(auto& iterator : Repo)
	{
		if (iterator.GetPrice() == PriceToFind)
		{
			toReturn.push_back(iterator);
		}
	}
	return toReturn;
}

void Repository::DeleteProduct(int IdToDelete)
{
	auto position = std::find_if(this->Repo.begin(), this->Repo.end(), [=](Product& productToDelete) noexcept {return productToDelete.GetId() == IdToDelete; });

	if (position == this->Repo.end())
	{
		throw std::string("Nu exista niciun produs cu acest id.\n");
	}

	this->Repo.erase(position);
}

std::vector<Product> Repository::GetAll() noexcept
{
	return std::vector<Product>(Repo);
}

int Repository::GetSize() noexcept
{
	return static_cast<int>(this->Repo.size());
}


void Bucket::add(Product& ToAdd)
{
	MemoryBucket.push_back(ToAdd);
}

void Bucket::clear()
{
	MemoryBucket.clear();
}

void Bucket::generate(int NumberOfProducts)
{
	std::mt19937 mt{ std::random_device{}() };
	std::uniform_int_distribution<> dist(0, MemoryBucket.size() - 1);
	int rndNr = dist(mt);// numar aleator intre [0,size-1]

	for (int count = 0; count < NumberOfProducts; count = count + 1)
	{
		MemoryBucket.push_back()
	}

}