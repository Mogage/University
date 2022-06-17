#include "service.h"
#include "exceptions.h"

Service::Service(Repository& _Repo) : 
	Repo{ _Repo }
{}

void Service::update(int Id, int Rank)
{
	if (Rank < 0 || Rank > 10)
		throw ValidationError("Rank invalid.\n");
	Repo.update(Id, Rank);
}

void Service::update(int Id, string Title)
{
	Repo.update(Id, Title);
}

void Service::deleteId(int Id)
{
	Repo.deleteId(Id);
}

Song Service::find(int Id)
{
	return Repo.findId(Id);
}

vector<Song> Service::all()
{
	return Repo.all();
}
