#include "userinterface.h"
#include "exceptions.h"
#include <iostream>
#include <fstream>

void UserInterface::PrintMenu()
{
	std::cout << "Menu:\n"
		<< "\t1.  Adauga element.\n"
		<< "\t2.  Modifica element.\n"
		<< "\t3.  Sterge element.\n"
		<< "\t4.  Afiseaza elementele.\n"
		<< "\t5.  Cauta produs.\n"
		<< "\t6.  Filtreaza produse.\n"
		<< "\t7.  Sorteaza produse.\n"
		<< "\t8.  Afiseaza meniu.\n"
		<< "\t9.  Adauga in cos.\n"
		<< "\t10. Goleste cos.\n"
		<< "\t11. Genereaza random in cos.\n"
		<< "\t12. Exporta cos.\n"
		<< "\t13. Iesire aplicatie.\n";
}

void UserInterface::Add()
{
	int intId = -1, intPrice = -1;
	std::string id, name, type, producer, price;

	while (true)
	{
		std::cout << "Introduceti codul de bare: ";
		std::getline(std::cin, id);
		if (id != "0" && 0 == (intId = atoi(id.c_str())))
		{
			std::cout << "Valoare invalida.\n";
			continue;
		}
		break;
	}

	std::cout << "Introdueti numele: ";
	std::getline(std::cin, name);

	std::cout << "Introdueti tipul: ";
	std::getline(std::cin, type);

	std::cout << "Introdueti producatorul: ";
	std::getline(std::cin, producer);

	while (true)
	{
		std::cout << "Introduceti pretul: ";
		std::getline(std::cin, price);
		if (price != "0" && 0 == (intPrice = atoi(price.c_str())))
		{
			std::cout << "Valoare invalida.\n";
			continue;
		}
		break;
	}
	

	try
	{
		this->Serv.AddProduct(intId, name, type, producer, intPrice);
		std::cout << "Produs adaugat cu succes.\n";
	}
	catch (ValidationError& error)
	{
		std::cout << error.getMessage();
	}
	catch (RepositoryError& error)
	{
		std::cout << error.getMessage();
	}
}

void UserInterface::Modify()
{
	int intId = -1, intPrice = -1;
	std::string strId, name, type, producer, strPrice;

	if (0 == this->Serv.GetAll().size())
	{
		std::cout << "Nu exista produse adaugate.\n";
		return;
	}

	while (true)
	{
		std::cout << "Introduceti codul de bare al produsului de modificat: ";
		std::getline(std::cin, strId);
		if (strId != "0" && 0 == (intId = atoi(strId.c_str())))
		{
			std::cout << "Valoare invalida.\n";
			continue;
		}
		break;
	}

	std::cout << "Introduceti noul nume sau enter pentru a nu il modifica: ";
	std::getline(std::cin, name);

	std::cout << "Introduceti noul tip sau enter pentru a nu il modifica: ";
	std::getline(std::cin, type);

	std::cout << "Introduceti noul producator sau enter pentru a nu il modifica: ";
	std::getline(std::cin, producer);

	while (true)
	{
		std::cout << "Introduceti noul pret sau enter pentru a nu il modifica: ";
		std::getline(std::cin, strPrice);
		if (strPrice != "" && strPrice != "0" && 0 == (intPrice = atoi(strPrice.c_str())))
		{
			std::cout << "Valoare invalida.\n";
			continue;
		}
		break;
	}

	try
	{
		this->Serv.ModifyProduct(intId, name, type, producer, intPrice);
		std::cout << "Produs modificat cu succes.\n";
	}
	catch (ValidationError& error)
	{
		std::cout << error.getMessage();
	}
	catch (RepositoryError& error)
	{
		std::cout << error.getMessage();
	}
}

void UserInterface::Delete()
{
	int intId = -1;
	std::string strId;

	if (0 == this->Serv.GetAll().size())
	{
		std::cout << "Nu exista produse adaugate.\n";
		return;
	}

	while (true)
	{
		std::cout << "Introduceti codul de bare al produsului de sters: ";
		std::getline(std::cin, strId);
		if (strId != "0" && 0 == (intId = atoi(strId.c_str())))
		{
			std::cout << "Valoare invalida.\n";
			continue;
		}
		break;
	}

	try
	{
		this->Serv.DeleteProduct(intId);
		std::cout << "Produs sters cu succes.\n";
	}
	catch (ValidationError& error)
	{
		std::cout << error.getMessage();
	}
	catch (RepositoryError& error)
	{
		std::cout << error.getMessage();
	}
}

void UserInterface::PrintAll()
{
	std::vector < Product > allProducts = this->Serv.GetAll();

	if (0 == allProducts.size())
	{
		std::cout << "Nu exista produse adaugate.\n";
		return;
	}

	for (const auto& iterator : allProducts)
	{
		std::cout << iterator.Print();
	}
}

void UserInterface::Find()
{
	int intId = -1;
	std::string strId;

	if (0 == this->Serv.GetAll().size())
	{
		std::cout << "Nu exista produse adaugate.\n";
		return;
	}

	while (true)
	{
		std::cout << "Introduceti codul de bare dupa care se cauta: ";
		std::getline(std::cin, strId);
		if (strId != "0" && 0 == (intId = atoi(strId.c_str())))
		{
			std::cout << "Valoare invalida.\n";
			continue;
		}
		break;
	}

	try
	{
		Product productToFind(this->Serv.FindProduct(intId));
		if (-1 == productToFind.GetId())
		{
			std::cout << "Produs inexistent.\n";
		}
		else
		{
			std::cout << productToFind.Print();
		}
	}
	catch (ValidationError& error)
	{
		std::cout << error.getMessage();
	}
	catch (RepositoryError& error)
	{
		std::cout << error.getMessage();
	}
}

void UserInterface::Filter()
{
	int intFilterType = -1, intPrice = -1;
	std::string strFilterType, name, producer, strPrice;

	if (0 == this->Serv.GetAll().size())
	{
		std::cout << "Nu exista produse adaugate.\n";
		return;
	}

	while (true)
	{
		std::cout << "Filtrati dupa: 1 - Nume, 2 - Producator, 3 - Pret: ";
		std::getline(std::cin, strFilterType);
		if (strFilterType != "0" && 0 == (intFilterType = atoi(strFilterType.c_str())))
		{
			std::cout << "Valoare invalida.\n";
			continue;
		}
		break;
	}

	try
	{
		std::vector < Product > products;
		switch (intFilterType)
		{
		case 1:
			std::cout << "Numele dupa care se filtreaza: ";
			std::getline(std::cin, name);
			products = this->Serv.Filter(name, 1);
			break;
		case 2:
			std::cout << "Producatorul dupa care se filtreaza: ";
			std::getline(std::cin, producer);
			products = this->Serv.Filter(producer, 2);
			break;
		case 3:
			while (true)
			{
				std::cout << "Pretul dupa care se filtreaza: ";
				std::getline(std::cin, strPrice);
				if (strPrice != "0" && 0 == (intPrice = atoi(strPrice.c_str())))
				{
					std::cout << "Valoare invalida.\n";
					continue;
				}
				break;
			}
			products = this->Serv.Filter(intPrice);
			break;
		default:
			std::cout << "Filtrul trebuie sa fie o valoare 1-3.\n";
			return;
		}
		if (0 == products.size())
		{
			std::cout << "Nu exista niciun produs cu acest filtru.\n";
		}
		else
		{
			for (auto& iterator : products)
			{
				std::cout << iterator.Print();
			}
		}
	}
	catch (ValidationError& error)
	{
		std::cout << error.getMessage();
	}
	catch (RepositoryError& error)
	{
		std::cout << error.getMessage();
	}
}

void UserInterface::Sort()
{
	int intSortType = -1;
	std::string strSortType;
	std::vector < Product > products;

	if (0 == this->Serv.GetAll().size())
	{
		std::cout << "Nu exista produse adaugate.\n";
		return;
	}

	while (true)
	{
		std::cout << "Sortati dupa: 1 - Nume, 2 - Pret, 3 - Nume si tip: ";
		std::getline(std::cin, strSortType);
		if (strSortType != "0" && 0 == (intSortType = atoi(strSortType.c_str())))
		{
			std::cout << "Valoare invalida.\n";
			continue;
		}
		break;
	}

	switch (intSortType)
	{
	case 1:
		products = this->Serv.Sort([](const Product& Product1, const Product& Product2) {return (Product1.GetName() < Product2.GetName()); });
		break;
	case 2:
		products = this->Serv.Sort([](const Product& Product1, const Product& Product2) noexcept {return (Product1.GetPrice() < Product2.GetPrice()); });
		break;
	case 3:
		products = this->Serv.Sort(
			[](const Product& Product1, const Product& Product2)
			{
				if (Product1.GetName() != Product2.GetName())
				{
					return (Product1.GetName() < Product2.GetName());
				}
				return (Product1.GetType() < Product2.GetType());
			}
		);
		break;
	default:
		std::cout << "Filtrul trebuie sa fie o valoare 1-3.\n";
		return;
	}
	for (const auto& iterator : products)
	{
		std::cout << iterator.Print();
	}

}

void UserInterface::AddBucket()
{
	int price;
	std::string name;

	if (0 == this->Serv.GetAll().size())
	{
		std::cout << "Nu exista produse adaugate.\n";
		return;
	}

	std::cout << "Numele produsului de adaugat in cos: ";
	std::getline(std::cin, name);

	try
	{
		price = ServBck.addToBucket(name);
		std::cout << "Valoare cos: " << price << "\n";
	}
	catch (ValidationError& error)
	{
		std::cout << error.getMessage();
	}
	catch (RepositoryError& error)
	{
		std::cout << error.getMessage();
	}
}

void UserInterface::ClearBucket()
{
	std::cout << "Valoare cos: " << ServBck.clearBucket() << "\n";
}

void UserInterface::GenerateBucket()
{
	int price, intNumberToGenerate = 0;
	std::string strNumberToGenerate;

	if (0 == this->Serv.GetAll().size())
	{
		std::cout << "Nu exista produse adaugate.\n";
		return;
	}

	while (true)
	{
		std::cout << "Numarul de produse de adaugat in cos: ";
		std::getline(std::cin, strNumberToGenerate);
		if (strNumberToGenerate != "0" && 0 == (intNumberToGenerate = atoi(strNumberToGenerate.c_str())))
		{
			std::cout << "Valoare invalida.\n";
			continue;
		}
		break;
	}

	price = ServBck.generateBucket(intNumberToGenerate);
	std::cout << "Valoare cos: " << price << "\n";
}

void UserInterface::ExportBucket()
{
	std::fstream fout;
	std::vector < Product > products = ServBck.getBucket();
	size_t length = products.size();
	std::string fileName = "";
	std::string path = "export_files/";
	Product toExport;

	while (fileName == "")
	{
		std::cout << "Numele fisierului pentru export: ";
		std::getline(std::cin, fileName);
	}

	path = path + fileName + ".csv";

	fout.open(path, std::ios::out, std::ios::trunc);
	fout.close();
	fout.open(path, std::ios::out | std::ios::app);

	for (size_t count = 0; count < length; count = count + 1)
	{
		toExport = products[count];
		fout << toExport.GetId() << ","
			<< toExport.GetName() << ","
			<< toExport.GetType() << ","
			<< toExport.GetProducer() << ","
			<< toExport.GetPrice() << "\n";
	}

	std::cout << "Cos exportat cu succes.\n";

	fout.close();
}

void UserInterface::Run()
{
	std::string input;

	this->PrintMenu();
	while (true)
	{
		std::cout << ">>>";
		std::getline(std::cin, input);

		if ("" == input)
		{
			continue;
		}

		switch (atoi(input.c_str()))
		{
		case 13:
			std::cout << "Aplicatie incheiata cu succes.\n";
			return;
		case 1:
			this->Add();
			break;
		case 2: 
			this->Modify();
			break;
		case 3:
			this->Delete();
			break;
		case 4:
			this->PrintAll();
			break;
		case 5:
			this->Find();
			break;
		case 6:
			this->Filter();
			break;
		case 7:
			this->Sort();
			break;
		case 8: 
			this->PrintMenu();
			break;
		case 9:
			this->AddBucket();
			break;
		case 10:
			this->ClearBucket();
			break;
		case 11:
			this->GenerateBucket();
			break;
		case 12:
			this->ExportBucket();
			break;
		default:
			std::cout << "Comanda invalida.\n";
			fseek(stdin, 0L, SEEK_END);
			break;
		}
	}

}
