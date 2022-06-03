#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <math.h>
#include <iomanip>

double Distanta(std::pair<int, int> Oras1, std::pair<int, int> Oras2)
{
	int difX = Oras2.first - Oras1.first;
	int difY = Oras2.second - Oras1.second;
	return std::sqrt(difX * difX + difY * difY);
}

class Muchie
{
public:
	int Sursa, Destinatie;
	double Cost;
	Muchie(int _Sursa, int _Dest, double _Cost) : Sursa{ _Sursa }, Destinatie{ _Dest }, Cost{ _Cost }{}
	bool operator<(Muchie& Other)
	{
		if (this->Cost != Other.Cost)
		{
			return (this->Cost < Other.Cost);
		}
		if (this->Sursa != Other.Sursa)
		{
			return (this->Sursa < Other.Sursa);
		}
		return (this->Destinatie < Other.Destinatie);
	}
};
class Set
{
private:
	int Size;
	std::vector < int > Rank, Parinte;
public:
	Set(int _Size) : Size{ _Size } 
	{
		Rank = std::vector<int>(Size, 0);
		Parinte = std::vector<int>(Size);
		for (int count = 0; count < Size; count = count + 1)
		{
			Parinte[count] = count;
		}
	}

	int find(int Nod)
	{
		if (Parinte[Nod] != Nod)
		{
			Parinte[Nod] = find(Parinte[Nod]);
		}

		return Parinte[Nod];
	}

	void union_sets(int Nod1, int Nod2)
	{
		int set1 = find(Nod1);
		int set2 = find(Nod2);

		if (Rank[set1] < Rank[set2])
		{
			Parinte[set1] = set2;
		}
		else
		{
			Parinte[set2] = set1;
			if (Rank[set2] == Rank[set1])
			{
				Rank[set1] = Rank[set1] + 1;
			}
		}
	}
};

double Kruskal(std::vector <Muchie> Muchii, int NrNoduri)
{
	double totalSum = 0;
	Set disjointSet(NrNoduri);

	std::sort(Muchii.begin(), Muchii.end());

	for (const auto& muchie : Muchii)
	{
		if (disjointSet.find(muchie.Sursa) != disjointSet.find(muchie.Destinatie))
		{
			totalSum = totalSum + muchie.Cost;
			disjointSet.union_sets(muchie.Sursa, muchie.Destinatie);
		}
	}

	return totalSum;
}

int main()
{
	int nrOrase, aux1, aux2;
	std::vector < std::pair < int, int > > coordonate;
	std::vector < Muchie > muchii;

	std::ifstream in("in.txt");

	in >> nrOrase;

	for (int oras = 0; oras < nrOrase; oras = oras + 1)
	{
		in >> aux1 >> aux2;
		coordonate.push_back({ aux1, aux2 });
	}

	in.close();

	aux1 = 0;
	for (auto& oras1 : coordonate)
	{
		aux2 = 0;
		for (auto& oras2 : coordonate)
		{
			muchii.push_back(Muchie{ aux1, aux2, Distanta(oras1, oras2) });
			aux2 = aux2 + 1;
		}
		aux1 = aux1 + 1;
	}

	std::ofstream out("out.txt");

	out << std::fixed << std::setprecision(3) << Kruskal(muchii, nrOrase);

	out.close();
	return 0;
}