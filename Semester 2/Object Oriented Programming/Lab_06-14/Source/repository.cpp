#include "repository.h"
#include "exceptions.h"
#include <fstream>
#include <sstream>
#include <random>

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
		if (iterator.GetName()._Starts_with(NameToFind) != false)
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
		if (iterator.GetProducer()._Starts_with(ProducerToFind) != false)
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

std::vector<Product> Repository::GetAll() const
{
	return std::vector<Product>(Repo);
}

int Repository::GetSize() const
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

void NewRepository::AruncaExceptie() const
{
	std::mt19937 mt{ std::random_device{}() };
	std::uniform_int_distribution<> dist(0, 100);
	double randomNumber = dist(mt);

	if (randomNumber / 100 <= Probability)
	{
		throw RepositoryError("Exceptie repo lab nou.\n");
	}
}

void NewRepository::AddProduct(Product& ProductToAdd)
{
	AruncaExceptie();
	for (auto& iterator : Repo)
	{
		if (iterator.second == ProductToAdd)
		{
			throw RepositoryError("Acest produs exista deja.\n");
		}
	}

	Repo.insert(std::make_pair(ProductToAdd.GetId(), ProductToAdd));
}

void NewRepository::DeleteProduct(int IdToDelete)
{
	AruncaExceptie();
	for (auto& iterator : Repo)
	{
		if (iterator.first == IdToDelete)
		{
			Repo.erase(IdToDelete);
			return;
			
		}
	}

	throw RepositoryError("Acest produs nu exista.\n");
}

void NewRepository::ModifyProduct(
	int			IdProductToModify,
	std::string NewName,
	std::string NewType,
	std::string NewProducer,
	int			NewPrice
	)
{
	AruncaExceptie();
	for (auto& iterator : Repo)
	{
		if (iterator.first == IdProductToModify)
		{
			throw RepositoryError("Acest produs nu exista.\n");
		}
	}

	if ("" != NewName)
	{
		Repo[IdProductToModify].SetName(NewName);
	}
	if ("" != NewType)
	{
		Repo[IdProductToModify].SetType(NewType);
	}
	if ("" != NewProducer)
	{
		Repo[IdProductToModify].SetProducer(NewProducer);
	}
	if (0 != NewPrice)
	{
		Repo[IdProductToModify].SetPrice(NewPrice);
	}
}

Product NewRepository::FindProductAfterID(int IdToFind) const
{
	AruncaExceptie();
	Product toReturn;
	for (auto& iterator : Repo)
	{
		if (iterator.first == IdToFind)
		{
			toReturn = iterator.second;
			break;
		}
	}
	return toReturn;
}

std::vector<Product> NewRepository::FindProductsAfterName(std::string NameToFind) const
{
	AruncaExceptie();
	std::vector< Product > toReturn;
	for (auto& iterator : Repo)
	{
		if (iterator.second.GetName() == NameToFind)
		{
			toReturn.push_back(iterator.second);
		}
	}
	return toReturn;
}

std::vector<Product> NewRepository::FindProductsAfterType(std::string TypeToFind) const
{
	AruncaExceptie();
	std::vector< Product > toReturn;
	for (auto& iterator : Repo)
	{
		if (iterator.second.GetType() == TypeToFind)
		{
			toReturn.push_back(iterator.second);
		}
	}
	return toReturn;
}

std::vector<Product> NewRepository::FindProductsAfterProducer(std::string ProducerToFind) const
{
	AruncaExceptie();
	std::vector< Product > toReturn;
	for (auto& iterator : Repo)
	{
		if (iterator.second.GetProducer() == ProducerToFind)
		{
			toReturn.push_back(iterator.second);
		}
	}
	return toReturn;
}

std::vector<Product> NewRepository::FindProductsAfterPrice(int PriceToFind) const
{
	AruncaExceptie();
	std::vector< Product > toReturn;
	for (auto& iterator : Repo)
	{
		if (iterator.second.GetPrice() == PriceToFind)
		{
			toReturn.push_back(iterator.second);
		}
	}
	return toReturn;
}

std::vector<Product> NewRepository::GetAll() const
{
	AruncaExceptie();
	std::vector < Product > toReturn;
	for (auto& iterator : Repo)
	{
		toReturn.push_back(iterator.second);
	}
	return toReturn;
}

int NewRepository::GetSize() const
{
	AruncaExceptie();
	return (int)Repo.size();
}
