#pragma once
#include "domain.h"
#include "repository.h"

/// <summary>
///		Clasa abstracta pentru undo
/// </summary>
class Undo
{
private:

public:
	/// <summary>
	///		Functia care face opusul ultimei operatii
	/// </summary>
	virtual void doUndo() = 0;

	/// <summary>
	///		Destructor virtual pentru undo
	/// </summary>
	virtual ~Undo() {}
};

class UndoAdd : public Undo
{
private:
	Repository& Repo;
	int Id;
public:
	/// <summary>
	///		Constructor pentru undo la adaugare
	/// </summary>
	/// <param name="Repo">Repo-ul cu produse din service</param>
	/// <param name="Id">Id-ul produsului adaugat</param>
	UndoAdd(Repository& Repo, int Id) : Repo{ Repo }, Id{ Id }{}

	/// <summary>
	///		Anuleaza operatia de adaugare produs
	/// </summary>
	void doUndo() override { Repo.DeleteProduct(Id); }
};

class UndoDelete : public Undo
{
private:
	Repository& Repo;
	Product DeletedProduct;
public:
	/// <summary>
	///		Constructor pentru undo la stergere
	/// </summary>
	/// <param name="Repo">Repo-ul cu produse din service</param>
	/// <param name="DeletedProduct">Produsul care a fost sters</param>
	UndoDelete(Repository& Repo, const Product& DeletedProduct) : Repo{ Repo }, DeletedProduct{ DeletedProduct }{}
	
	/// <summary>
	///		Anuleaza operatia de stergere produs
	/// </summary>
	void doUndo() override { Repo.AddProduct(DeletedProduct); }
};

class UndoModify : public Undo
{
private:
	Repository& Repo;
	Product OldProduct;
public:
	/// <summary>
	///		Constructor pentru undo la modificare
	/// </summary>
	/// <param name="Repo">Repo-ul cu produse din service</param>
	/// <param name="OldProduct">Produsul care a fost modificat</param>
	UndoModify(Repository& Repo, const Product& OldProduct) : Repo{ Repo }, OldProduct{ OldProduct }{}
	
	/// <summary>
	///		Anuleaza operatia de modificare produs
	/// </summary>
	void doUndo() override { Repo.DeleteProduct(OldProduct.GetId()); Repo.AddProduct(OldProduct); }
};