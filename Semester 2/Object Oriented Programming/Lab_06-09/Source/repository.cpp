#include "repository.h"
#include "exceptions.h"
#include <fstream>
#include <sstream>

void Repository::AddProduct(Product& ProductToAdd)
{
	for(auto& iterator : Repo)
	{
		if (iterator == ProductToAdd)
		{
			throw RepositoryError("Acest produs exista deja.\n");
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
		throw RepositoryError("Nu exista niciun produs cu acest id.\n");
	}
}

Product Repository::FindProductAfterID(int IdToFind) const 
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

std::vector < Product > Repository::FindProductsAfterName(std::string NameToFind) const 
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

std::vector < Product > Repository::FindProductsAfterType(std::string TypeToFind) const 
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

std::vector < Product > Repository::FindProductsAfterProducer(std::string ProducerToFind) const
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

std::vector < Product > Repository::FindProductsAfterPrice(int PriceToFind) const 
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
		throw RepositoryError("Nu exista niciun produs cu acest id.\n");
	}

	this->Repo.erase(position);
}

std::vector<Product> Repository::GetAll() const noexcept
{
	return std::vector<Product>(Repo);
}

int Repository::GetSize() const noexcept
{
	return static_cast<int>(this->Repo.size());
}

int Bucket::add(Product& ToAdd)
{
	MemoryBucket.push_back(ToAdd);
	TotalPrice = TotalPrice + ToAdd.GetPrice();
	return TotalPrice;
}

int Bucket::clear()
{
	MemoryBucket.clear();
	TotalPrice = 0;
	return TotalPrice;
}

const std::vector < Product >& Bucket::getBucket() const
{
	return MemoryBucket;
}

int Bucket::getPrice() const
{
	return TotalPrice;
}

void FileRepository::LoadFromFile()
{
	std::ifstream in(FilePath);
	std::string line, name, type, producer;
	int id, price;

	while (std::getline(in, line))
	{
		std::istringstream stringInput(line);
		stringInput >> id;
		std::getline(stringInput, name, ',');
		std::getline(stringInput, name, ',');
		std::getline(stringInput, type, ',');
		std::getline(stringInput, producer, ',');
		stringInput >> price;

		Product toAdd{ id, name, type, producer, price };
		Repository::AddProduct(toAdd);
	}

	in.close();
}

void FileRepository::WriteToFile()
{
	std::vector < Product > products = Repository::GetAll();
	std::ofstream out(FilePath);

	if (out.is_open())
	{
		for (const auto& product : products)
		{
			out << product.GetId() << ","
				<< product.GetName() << ","
				<< product.GetType() << ","
				<< product.GetProducer() << ","
				<< product.GetPrice() << "\n";
		}
	}

	out.close();
}

FileRepository::FileRepository(std::string FileName) : Repository()
{
	FilePath = "save_files/" + FileName;
	LoadFromFile();
}

void FileRepository::AddProduct(Product& ProductToAdd)
{
	Repository::AddProduct(ProductToAdd);
	WriteToFile();
}

void FileRepository::ModifyProduct(int IdProductToModify, std::string NewName, std::string NewType, std::string NewProducer, int NewPrice)
{
	Repository::ModifyProduct(IdProductToModify, NewName, NewType, NewProducer, NewPrice);
	WriteToFile();
}

void FileRepository::DeleteProduct(int IdToDelete)
{
	Repository::DeleteProduct(IdToDelete);
	WriteToFile();
}