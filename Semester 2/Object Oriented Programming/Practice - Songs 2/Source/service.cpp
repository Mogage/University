#include "service.h"
#include <unordered_set>

Service::Service(Repository& _Repo) : Repo{ _Repo }
{}

int findSmallestMissing(const vector< Song >& Songs)
{
	std::unordered_set<int> distinct;

	for (const auto& song : Songs)
	{
		distinct.insert(song.id());
	}

	int index = 1;
	while (true)
	{
		if (distinct.find(index) == distinct.end()) {
			return index;
		}
		index++;
	}
}

void Service::add(string Title, string Artist, string Gen)
{
	int newId = findSmallestMissing(Repo.all());
	Song toAdd{ newId , Title, Artist, Gen };
	Repo.add(toAdd);
}

void Service::del(int Id)
{
	Repo.deleteId(Id);
}

vector<Song> Service::all()
{
	return Repo.all();
}
