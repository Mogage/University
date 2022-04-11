#include <iostream>
#include <fstream>
#include <vector>
#include <queue>
#include <string>
#include <assert.h>

constexpr auto INF = 2000000000;

std::vector < int > Dijkstra(std::vector < std::vector < std::pair < int, int > > > ListaAdiacenta, int Sursa)
{
	int nrVarfuri = (int)ListaAdiacenta.size() - 1;
	std::pair < int, int > aux;
	std::vector < int > distante(nrVarfuri, INF);
	std::priority_queue < std::pair < int, int > > pq;

	distante[Sursa] = 0;
	pq.push({ 0, Sursa });

	while (false == pq.empty())
	{
		aux = pq.top();
		pq.pop();
		aux.first = -aux.first;
		if (aux.first > distante[aux.second])
		{
			continue;
		}
		for (const auto& iterator : ListaAdiacenta[aux.second])
		{

			if (distante[iterator.first] > distante[aux.second] + iterator.second)
			{
				distante[iterator.first] = distante[aux.second] + iterator.second;
				pq.push(std::make_pair(-distante[iterator.first], iterator.first));
			}
		}
	}

	return distante;
}

void Problema1(std::string InFile, std::string OutFile)
{
	std::ifstream fin(InFile);
	std::ofstream fout(OutFile);

	int varf1, varf2, cost;
	int numarVarfuri, numarMuchii, varfSursa;
	std::vector < std::vector < std::pair < int, int > > > listaAdiacenta;
	std::vector < int > rezultat;

	fin >> numarVarfuri >> numarMuchii >> varfSursa;

	listaAdiacenta = std::vector < std::vector < std::pair < int, int > > >(numarVarfuri + 1);

	for (int indice = 0; indice < numarMuchii; indice = indice + 1)
	{
		fin >> varf1 >> varf2 >> cost;
		listaAdiacenta[varf1].push_back(std::make_pair(varf2, cost));
	}

	rezultat = Dijkstra(listaAdiacenta, varfSursa);

	for (const auto& iterator : rezultat)
	{
		if (iterator == INF)
		{
			fout << "INF ";
		}
		else
		{
			fout << iterator << " ";
		}
	}

	fin.close();
	fout.close();
}

void Test1()
{
	std::string inFile;
	std::string outFile;
	std::string outData, auxData;

	std::cout << "------Teste problema 1 ------ \n\n";

	for (int i = 1; i <= 7; i = i + 1)
	{
		std::cout << i << " ";

		inFile = "1/" + std::to_string(i) + "-in.txt";
		outFile = "1/" + std::to_string(i) + "-out.txt";

		Problema1(inFile, "1/aux-out.txt");

		std::ifstream toCheck(outFile);
		std::ifstream auxFile("1/aux-out.txt");

		std::getline(toCheck, outData);
		std::getline(auxFile, auxData);

		outData.erase(std::remove_if(outData.begin(), outData.end(), isspace), outData.end());
		auxData.erase(std::remove_if(auxData.begin(), auxData.end(), isspace), auxData.end());

		assert(outData == auxData);

		auxFile.close();
		toCheck.close();
	}

	std::cout << "\t7/7";
}

bool Relaxare(std::vector < std::pair < int, int > >& Distante, int Varf, int VarfVecin, int Cost)
{
	if (Distante[VarfVecin].first > Distante[Varf].first + Cost)
	{
		Distante[VarfVecin].first = Distante[Varf].first + Cost;
		Distante[VarfVecin].second = Varf;
		return true;
	}

	return false;
}

bool BellmanFord(std::vector < std::vector < std::pair < int, int > > > ListaAdiacenta, int Sursa, std::vector < std::pair < int, int > >& Distante)
{
	int nrVarfuri = (int)ListaAdiacenta.size() - 1;
	/// .first - drum ; .second - parinte
	std::vector < std::pair < int, int > > distante(nrVarfuri + 1, std::make_pair(INF, -1));

	distante[Sursa].first = 0;

	for (int i = 0; i < nrVarfuri - 1; i = i + 1)
	{
		for (int j = 0; j < nrVarfuri; j = j + 1)
		{
			for (const auto& iterator : ListaAdiacenta[j])
			{
				Relaxare(distante, j, iterator.first, iterator.second);
			}
		}
	}

	for (int i = 0; i < nrVarfuri; i = i + 1)
	{
		for (const auto& iterator : ListaAdiacenta[i])
		{
			if (Relaxare(distante, i, iterator.first, iterator.second))
			{
				return false;
			}
		}
	}

	Distante = distante;
	return true;
}

void Johnson(std::vector < std::vector < std::pair < int, int > > > ListaAdiacenta, std::string OutFile)
{
	int nrVarfuri = (int)ListaAdiacenta.size() - 1;
	int varfNou = nrVarfuri;
	std::vector < std::vector < std::pair < int, int > > > ListaAdiacentaJohnson(nrVarfuri + 1);
	std::vector < int > h(nrVarfuri + 1);
	std::vector < std::pair < int, int > > distante;
	std::vector < int > rezultat;

	std::ofstream fout(OutFile);

	for (int index = 0; index < nrVarfuri; index = index + 1)
	{
		ListaAdiacentaJohnson[varfNou].push_back(std::make_pair(index, 0));
		for (const auto& iterator : ListaAdiacenta[index])
		{
			ListaAdiacentaJohnson[index].push_back(iterator);
		}
	}
	
	if (false == BellmanFord(ListaAdiacentaJohnson, varfNou, distante))
	{
		fout << -1;
		fout.close();
		return;
	}

	for (int index = 0; index <= nrVarfuri; index = index + 1)
	{
		h[index] = distante[index].first;
	}

	for (int index = 0; index <= nrVarfuri; index = index + 1)
	{
		for (auto& iterator : ListaAdiacentaJohnson[index])
		{
			iterator.second = iterator.second + h[index] - h[iterator.first];
		}
	}
	
	for (int index = 0; index < nrVarfuri; index = index + 1)
	{
		for (auto& iterator : ListaAdiacentaJohnson[index])
		{
			fout << index << " " << iterator.first << " " << iterator.second << "\n";
		}
	}

	for (int index = 0; index < nrVarfuri; index = index + 1)
	{
		rezultat = Dijkstra(ListaAdiacenta, index);

		for (const auto& iterator : rezultat)
		{
			if (iterator == INF)
			{
				fout << "INF ";
			}
			else
			{
				fout << iterator << " ";
			}
		}
		fout << "\n";
	}

	fout.close();
}

void Problema2(std::string InFile, std::string OutFile)
{
	std::ifstream fin(InFile);

	int varf1, varf2, cost;
	int numarVarfuri, numarMuchii;
	std::vector < std::vector < std::pair < int, int > > > listaAdiacenta;
	std::vector < int > rezultat;

	fin >> numarVarfuri >> numarMuchii;

	listaAdiacenta = std::vector < std::vector < std::pair < int, int > > >(numarVarfuri + 1);

	for (int indice = 0; indice < numarMuchii; indice = indice + 1)
	{
		fin >> varf1 >> varf2 >> cost;
		listaAdiacenta[varf1].push_back(std::make_pair(varf2, cost));
	}

	Johnson(listaAdiacenta, OutFile);

	fin.close();
}

void Test2()
{
	std::string inFile;
	std::string outFile;
	std::string outData, auxData;

	std::cout << "\n\n------Teste problema 2 ------ \n\n";

	for (int i = 1; i <= 10; i = i + 1)
	{
		std::cout << i << " ";

		inFile = "2/" + std::to_string(i) + "-in.txt";
		outFile = "2/" + std::to_string(i) + "-out.txt";

		Problema2(inFile, "2/aux-out.txt");

		std::ifstream toCheck(outFile);
		std::ifstream auxFile("2/aux-out.txt");


		while (std::getline(toCheck, outData))
		{
			std::getline(auxFile, auxData);

			outData.erase(std::remove_if(outData.begin(), outData.end(), isspace), outData.end());
			auxData.erase(std::remove_if(auxData.begin(), auxData.end(), isspace), auxData.end());

			assert(outData == auxData);
		}
		
		auxFile.close();
		toCheck.close();
	}

	std::cout << "\t10/10";
}

int main(int argc, char * argv[])
{
	Test1();

	Test2();

	return 0;
}