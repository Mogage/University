#include "test.h"
#include "domain.h"
#include "repository.h"
#include "service.h"
#include <assert.h>

void Test::TestProduct()
{
	Product emptyProduct; // 1
	Product dataProduct1(1, "lapte", "bautura", "dorna", 7); // 2
	Product dataProduct2(2, "lapte", "bautura", "dorna", 7); // 3

	assert(dataProduct1 == dataProduct2);
	assert(false == (dataProduct1 != dataProduct2));
	assert(false == (dataProduct1 != dataProduct1));
	assert(dataProduct1 != emptyProduct);
	assert(false == (dataProduct1 == emptyProduct));

	assert(dataProduct1.GetId() == 1);
	assert(dataProduct1.GetName() == "lapte");
	assert(dataProduct1.GetType() == "bautura");
	assert(dataProduct1.GetProducer() == "dorna");
	assert(dataProduct1.GetPrice() == 7);

	dataProduct2.SetName("cartofi");
	dataProduct2.SetType("snacks");
	dataProduct2.SetProducer("lays");
	dataProduct2.SetPrice(5);

	assert(dataProduct1 != dataProduct2);
	assert(dataProduct2.Print() == "Id: 2, Nume: cartofi, Tip: snacks, Producator: lays, Pret: 5.\n");
	assert(emptyProduct.Print() == "Produs invalid.\n");
}

// -------------------------------------------------------------------------------------------

void Test::TestRepoAdd()
{
	Repository repo;
	Product product1(1, "lapte", "bautura", "dorna", 7); // 4
	Product product2(2, "cartofi", "snacks", "lays", 4); // 5
	std::vector < Product > result;

	assert(repo.GetSize() == 0);
	assert(repo.GetAll().size() == 0);

	repo.AddProduct(product1);
	repo.AddProduct(product2);
	result = repo.GetAll();

	assert(repo.GetSize() == 2);
	assert(result[0] == product1);
	assert(result[1] == product2);
	assert(result[1].Print() == "Id: 2, Nume: cartofi, Tip: snacks, Producator: lays, Pret: 4.\n");

	try
	{
		repo.AddProduct(product1);
	}
	catch (const std::string error)
	{
		assert(error == "Acest produs exista deja.\n");
	}
}

void Test::TestRepoUpdate()
{
	Repository repo;
	Product product1(1, "lapte", "bautura", "dorna", 7); // 6
	Product product2(2, "cartofi", "snacks", "lays", 4); // 7
	std::vector < Product > result;

	assert(repo.GetSize() == 0);
	assert(repo.GetAll().size() == 0);

	repo.AddProduct(product1);
	repo.AddProduct(product2);
	repo.ModifyProduct(1, "vin", "alcool", "", 30);
	repo.ModifyProduct(2, "", "", "chio");
	result = repo.GetAll();

	assert(result[0].GetName() == "vin");
	assert(result[0].GetType() == "alcool");
	assert(result[0].GetProducer() == "dorna");
	assert(result[0].GetPrice() == 30);
	assert(result[1].GetName() == "cartofi");
	assert(result[1].GetType() == "snacks");
	assert(result[1].GetProducer() == "chio");
	assert(result[1].GetPrice() == 4);

	try
	{
		repo.ModifyProduct(3, "ana");
	}
	catch (const std::string error)
	{
		assert(error == "Nu exista niciun produs cu acest id.\n");
	}
}

void Test::TestRepoFind()
{
	Repository repo;
	Product emptyProduct; // 8
	Product product1(1, "lapte", "bautura", "dorna", 7); // 9
	Product product2(2, "cartofi", "snacks", "lays", 4); // 10
	Product product3(3, "cartofi", "alcool", "dorna", 6); // 11
	Product product4(4, "vin", "alcool", "chio", 7); // 12
	std::vector < Product > result1, result2, result3, result4;

	assert(repo.GetSize() == 0);
	assert(repo.GetAll().size() == 0);

	repo.AddProduct(product1);
	repo.AddProduct(product2);
	repo.AddProduct(product3);
	repo.AddProduct(product4);
	result1 = repo.FindProductsAfterName("cartofi");
	// repo.FindProductsAfterName("cartofi", result1);
	result2 = repo.FindProductsAfterType("alcool");
	result3 = repo.FindProductsAfterProducer("dorna");
	result4 = repo.FindProductsAfterPrice(7);

	assert(emptyProduct == repo.FindProductAfterID(10)); // 55
	assert(product1 == repo.FindProductAfterID(1)); // 56
	assert(product2 == repo.FindProductAfterID(2)); // 57
	assert(result1.size() == 2);
	assert(result2.size() == 2);
	assert(result3.size() == 2);
	assert(result4.size() == 2);
	assert(result1[0] == product2);
	assert(result1[1] == product3);
	assert(result2[0] == product3);
	assert(result2[1] == product4);
	assert(result3[0] == product1);
	assert(result3[1] == product3);
	assert(result4[0] == product1);
	assert(result4[1] == product4);
}

void Test::TestRepoDelete()
{
	Repository repo;
	Product product1(1, "lapte", "bautura", "dorna", 7); // 13
	Product product2(2, "cartofi", "snacks", "lays", 4); // 14
	Product product3(3, "cartofi", "alcool", "dorna", 6); // 15
	Product product4(4, "vin", "alcool", "chio", 7); // 16
	std::vector < Product > result1, result2, result3, result4;

	assert(repo.GetSize() == 0);
	assert(repo.GetAll().size() == 0);

	repo.AddProduct(product1);
	repo.AddProduct(product2);
	repo.AddProduct(product3);
	repo.AddProduct(product4);

	repo.DeleteProduct(1);

	assert(repo.GetSize() == 3);
	assert(repo.GetAll().size() == 3);

	repo.DeleteProduct(4);

	assert(repo.GetSize() == 2);
	assert(repo.GetAll().size() == 2);

	try
	{
		repo.DeleteProduct(1);
	}
	catch (const std::string error)
	{
		assert(error == "Nu exista niciun produs cu acest id.\n");
	}
}

void Test::TestRepository()
{
	this->TestRepoAdd();
	this->TestRepoUpdate();
	this->TestRepoFind();
	this->TestRepoDelete();
}

// -------------------------------------------------------------------------------------------

void Test::TestValidator()
{
	Validator valid;
	Product emptyProduct; // 17
	Product dataProduct1(1, "lapte", "bautura", "", -25); // 18
	Product dataProduct2(2, "", "", "dorna", 7); //19
	Product dataProduct3(3, "lapte", "bautura", "dorna", 7); //20

	try
	{
		valid.ValidateProduct(emptyProduct);
	}
	catch (const std::string errors)
	{
		assert(errors == "Id-ul produsului este invalid.\nNumele produsului este invalid.\nTipul produsului este invalid.\nProducatorul produsului este invalid.\nPretul produsului este invalid.\n");
	}

	try
	{
		valid.ValidateProduct(dataProduct1);
	}
	catch (const std::string errors)
	{
		assert(errors == "Producatorul produsului este invalid.\nPretul produsului este invalid.\n");
	}

	try
	{
		valid.ValidateProduct(dataProduct2);
	}
	catch (const std::string errors)
	{
		assert(errors == "Numele produsului este invalid.\nTipul produsului este invalid.\n");
	}

	valid.ValidateProduct(dataProduct3);
}

// -------------------------------------------------------------------------------------------

void Test::TestServAdd()
{
	Repository repo;
	Validator valid;
	Service serv(repo, valid);
	Product productToCheck(1, "lapte", "bautura", "dorna", 7); // 21
	std::vector < Product > result;

	serv.AddProduct(1, "lapte", "bautura", "dorna", 7); // 22
	serv.AddProduct(2, "cartofi", "snacks", "lays", 4); // 23

	try
	{
		serv.AddProduct(1, "lapte", "bautura", "dorna", 7); // 24
	}
	catch (const std::string error)
	{
		assert(error == "Acest produs exista deja.\n");
	}

	try
	{
		serv.AddProduct(-5, "", "snacks", "chio", -1005); // 25
	}
	catch (const std::string error)
	{
		assert(error == "Id-ul produsului este invalid.\nNumele produsului este invalid.\nPretul produsului este invalid.\n");
	}

	result = serv.GetAll();

	assert(result.size() == 2);
	assert(repo.FindProductAfterID(1) == productToCheck);

}

void Test::TestServUpdate()
{
	Repository repo;
	Validator valid;
	Service serv(repo, valid);
	Product productToCheck(1, "lapte", "bautura", "dorna", 7); //26
	std::vector < Product > result;

	serv.AddProduct(1, "lapte", "bautura", "dorna", 7); //27
	serv.AddProduct(2, "cartofi", "snacks", "lays", 4); //28

	try
	{
		serv.ModifyProduct(-1);
	}
	catch (const std::string error)
	{
		assert(error == "Id invalid.\n");
	}
	try
	{
		serv.ModifyProduct(5, "", "", "", -25);
	}
	catch (const std::string error)
	{
		assert(error == "Pret invalid.\n");
	}

	serv.ModifyProduct(1);

	assert(serv.GetAll()[0] == productToCheck);

	serv.ModifyProduct(1, "", "-", "", 20);
	serv.ModifyProduct(2, "alune", "", "nestle");

	assert(serv.GetAll()[0].GetName() == "lapte");
	assert(serv.GetAll()[0].GetType() == "-");
	assert(serv.GetAll()[0].GetProducer() == "dorna");
	assert(serv.GetAll()[0].GetPrice() == 20);
	assert(serv.GetAll()[1].GetName() == "alune");
	assert(serv.GetAll()[1].GetType() == "snacks");
	assert(serv.GetAll()[1].GetProducer() == "nestle");
	assert(serv.GetAll()[1].GetPrice() == 4);
}

void Test::TestServFind()
{
	Repository repo;
	Validator valid;
	Service serv(repo, valid);
	Product product1(1, "lapte", "bautura", "dorna", 7); // 29
	Product product2(2, "cartofi", "snacks", "lays", 4); // 30
	Product product3(3, "cartofi", "alcool", "dorna", 6); // 31
	Product product4(4, "vin", "alcool", "chio", 8); // 32
	std::vector < Product > result;

	serv.AddProduct(1, "lapte", "bautura", "dorna", 7); // 33
	serv.AddProduct(2, "cartofi", "snacks", "lays", 4); // 34
	serv.AddProduct(3, "cartofi", "alcool", "dorna", 6); // 35
	serv.AddProduct(4, "vin", "alcool", "chio", 8); // 36

	assert(product1 == serv.FindProduct(1)); // 58

	/*result = serv.FindProduct(1);
	assert(result.size() == 1);
	assert(result[0] == product1);

	result = serv.FindProduct("cartofi", 2);
	assert(result.size() == 2);
	assert(result[0] == product2);
	assert(result[1] == product3);

	result = serv.FindProduct("bautura", 3);
	assert(result.size() == 1);
	assert(result[0] == product1);

	result = serv.FindProduct("dorna", 4);
	assert(result.size() == 2);
	assert(result[0] == product1);
	assert(result[1] == product3);

	result = serv.FindProduct(8, 5);
	assert(result.size() == 1);
	assert(result[0] == product4);
	*/
	try { serv.FindProduct(-1); } // 59
	catch (const std::string error) { assert(error == "Cod de bare invalid.\n"); }
	/*try { serv.FindProduct(-1, 5); }
	catch (const std::string error) { assert(error == "Pret invalid.\n"); }
	try { serv.FindProduct("", 2); }
	catch (const std::string error) { assert(error == "Nume invalid.\n"); }
	try { serv.FindProduct("", 3); }
	catch (const std::string error) { assert(error == "Tip invalid.\n"); }
	try { serv.FindProduct("", 4); }
	catch (const std::string error) { assert(error == "Producator invalid.\n"); }
	try { serv.FindProduct(2, 2); }
	catch (const std::string error) { assert(error == "int - FieldToSearch: Invalid value(1/5).\n"); }
	try { serv.FindProduct("bautura", 1); }
	catch (const std::string error) { assert(error == "string - FieldToSearch: Invalid value(2-4).\n"); }
	*/
}

void Test::TestServDelete()
{
	Repository repo;
	Validator valid;
	Service serv(repo, valid);

	serv.AddProduct(1, "lapte", "bautura", "dorna", 7); // 37
	serv.AddProduct(2, "cartofi", "snacks", "lays", 4); // 38

	serv.DeleteProduct(2);

	try
	{
		serv.DeleteProduct(-25);
	}
	catch (const std::string error)
	{
		assert(error == "Id invalid.\n");
	}
	try
	{
		serv.DeleteProduct(2);
	}
	catch (const std::string error)
	{
		assert(error == "Nu exista niciun produs cu acest id.\n");
	}

	assert(serv.GetAll().size() == 1);
}

void Test::TestFilter()
{
	Repository repo;
	Validator valid;
	Service serv(repo, valid);
	Product product1(1, "lapte", "bautura", "dorna", 7); // 39
	Product product2(2, "cartofi", "snacks", "lays", 4); // 40
	Product product3(3, "cartofi", "alcool", "dorna", 6); // 41
	Product product4(4, "vin", "alcool", "chio", 7); // 42
	std::vector < Product > result;

	serv.AddProduct(1, "lapte", "bautura", "dorna", 7); // 43
	serv.AddProduct(2, "cartofi", "snacks", "lays", 4); // 44
	serv.AddProduct(3, "cartofi", "alcool", "dorna", 6); // 45
	serv.AddProduct(4, "vin", "alcool", "chio", 7); // 46

	result = serv.Filter(7);
	assert(result.size() == 2);
	assert(result[0] == product1);
	assert(result[1] == product4);

	result = serv.Filter("cartofi", 1);
	assert(result.size() == 2);
	assert(result[0] == product2);
	assert(result[1] == product3);

	result = serv.Filter("dorna", 2);
	assert(result.size() == 2);
	assert(result[0] == product1);
	assert(result[1] == product3);

	assert(serv.Filter("dorna", 1).size() == 0);

	try { serv.Filter(-25); }
	catch (const std::string error) { assert(error == "Pret invalid.\n"); }
	try { serv.Filter("", 1); }
	catch (const std::string error) { assert(error == "Date invalide.\n"); }
	try { serv.Filter("dorna", 3); }
	catch (const std::string error) { assert(error == "Tip filtru invalid.\n"); }
}

void Test::TestSort()
{
	Repository repo;
	Validator valid;
	Service serv(repo, valid);
	Product product1(1, "lapte", "bautura", "dorna", 8); // 47
	Product product2(2, "cartofi", "snacks", "lays", 4); // 48
	Product product3(3, "cartofi", "alcool", "dorna", 6); // 49
	Product product4(4, "vin", "alcool", "chio", 7); // 50
	std::vector < Product > result;

	serv.AddProduct(1, "lapte", "bautura", "dorna", 8); // 51
	serv.AddProduct(2, "cartofi", "snacks", "lays", 4); // 52
	serv.AddProduct(3, "cartofi", "alcool", "dorna", 6); // 53
	serv.AddProduct(4, "vin", "alcool", "chio", 7); // 54

	result = serv.Sort([](const Product& Product1, const Product& Product2) noexcept {return (Product1.GetPrice() < Product2.GetPrice()); });
	assert(result.size() == 4);
	assert(result[0] == product2);
	assert(result[1] == product3);
	assert(result[2] == product4);
	assert(result[3] == product1);

	result = serv.Sort([](const Product& Product1, const Product& Product2) {return (Product1.GetName() < Product2.GetName()); });
	assert(result.size() == 4);
	assert(result[0] == product2);
	assert(result[1] == product3);
	assert(result[2] == product1);
	assert(result[3] == product4);

	result = serv.Sort(
		[](const Product& Product1, const Product& Product2)
		{
			if (Product1.GetName() != Product2.GetName())
			{
				return (Product1.GetName() < Product2.GetName());
			}
			return (Product1.GetType() < Product2.GetType());
		}
		);
	assert(result.size() == 4);
	assert(result[0] == product3);
	assert(result[1] == product2);
	assert(result[2] == product1);
	assert(result[3] == product4);
}

void Test::TestService()
{
	this->TestServAdd();
	this->TestServUpdate();
	this->TestServFind();
	this->TestServDelete();
	this->TestFilter();
	this->TestSort();
}

void Test::RunAll()
{
	this->TestProduct();
	this->TestRepository();
	this->TestValidator();
	this->TestService();
}