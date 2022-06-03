#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <queue>
#include <list>
#include <stack>
#include <assert.h>

int Bfs(int Sursa, int Dest, std::vector< int >& Parent, std::vector < std::vector < int > > Graf);
int EdmondsKarp(std::vector < std::vector < std::pair < int, int > > > Graf, int Sursa, int Dest);

class Nod
{
public:
	int NumarNod, Inaltime, Exces;
	std::vector < int > Vecini;
	std::vector < int >::iterator Curent;

	Nod() { Exces = 0; Inaltime = 0; NumarNod = 0; }
	Nod(int _NumarNod) : NumarNod{ _NumarNod } { Exces = 0; Inaltime = 0; }
};

void InitializarePreflux(std::vector < std::vector < std::pair < int, int > > >& Graf, std::vector < Nod* >& Retea, Nod*& Sursa);
void Pompare(std::vector < std::vector < std::pair < bool, int > > >& Graf, Nod*& DeTrimis, Nod*& DePrimit);
void Descarca(std::vector < std::vector < std::pair < int, int > > >& Graf, std::vector < Nod* >& Retea, Nod*& DeDescarcat);
void Inaltare(std::vector < std::vector < std::pair < int, int > > >& Graf, std::vector < Nod* >& Retea, Nod*& DeInaltat);
int PompareTopologica(std::vector < std::vector < std::pair < int, int > > > Graf, int Sursa, int Dest);

std::vector < std::pair < int, int > > Muchii;
std::vector < int > Ciclu;
std::vector < bool > Elim;
std::vector < std::vector < int > > G;
void Euler()
{
	int top, val, next;
	std::stack < int > stiva;
	stiva.push(0);

	while (!stiva.empty())
	{
		top = stiva.top();
		if (G[top].size())
		{
			val = G[top].back();
			G[top].pop_back();
			if (!Elim[val])
			{
				Elim[val] = true;
				next = Muchii[val].second;
				if (next == top)
					next = Muchii[val].first;
				stiva.push(next);
			}
		}
		else
		{
			Ciclu.push_back(top);
			stiva.pop();
		}
	}
}

void Problema1(std::string InFile, std::string OutFile);
void Problema2(std::string InFile, std::string OutFile);
void Problema3(std::string InFile, std::string OutFile);

void Test(int NrProblema, int NrTeste);

int main()
{
	//Test(1, 10);
	//Test(2, 10);
	//Test(3, 9);

	Problema2("2/5-in.txt", "2/aux-out.txt");
	return 0;
}

int Bfs(int Sursa, int Dest, std::vector< int >& Parent, std::vector < std::vector < int > > Graf)
{
	int current, flow, newFlow;
	std::queue < std::pair < int, int > > bfsQueue;

	std::fill(Parent.begin(), Parent.end(), -1);
	Parent[Sursa] = -2;
	
	bfsQueue.push({ Sursa, INT_MAX });
	
	while (false == bfsQueue.empty())
	{
		current = bfsQueue.front().first;
		flow = bfsQueue.front().second;
		bfsQueue.pop();

		for (int vecin = 0; vecin < (int)Graf.size(); vecin = vecin + 1)
		{
			if (-1 == Parent[vecin] && Graf[current][vecin])
			{
				Parent[vecin] = current;
				newFlow = (flow < Graf[current][vecin]) ? flow : Graf[current][vecin];
				if (Dest == vecin)
					return newFlow;
				bfsQueue.push({ vecin, newFlow });
			}
		}
	}
	return 0;
}

int EdmondsKarp(std::vector < std::vector < std::pair < int, int > > > Graf, int Sursa, int Dest)
{
	int flow = 0, newFlow, nrNoduri = (int)Graf.size(), current, prev;
	std::vector < std::vector < int > > rGraf(nrNoduri, std::vector < int >(nrNoduri, 0));
	std::vector < int > parent(nrNoduri);

	for (int nod = 0; nod < nrNoduri; nod = nod + 1)
	{
		for (auto vecin : Graf[nod])
		{
			rGraf[nod][vecin.first] = vecin.second;
		}
	}

	while (newFlow = Bfs(Sursa, Dest, parent, rGraf))
	{
		flow = flow + newFlow;
		current = Dest;
		while (Sursa != current)
		{
			prev = parent[current];
			rGraf[prev][current] = rGraf[prev][current] - newFlow;
			rGraf[current][prev] = rGraf[current][prev] + newFlow;
			current = prev;
		}
	}

	return flow;
}

void InitializarePreflux(std::vector < std::vector < std::pair < int, int > > >& Graf, std::vector < Nod* >& Retea, Nod*& Sursa)
{
	int nrNoduri = (int)Graf.size();

	Sursa->Inaltime = nrNoduri;
	for (int nod = 0; nod < nrNoduri; nod = nod + 1)
	{
		if (Graf[Sursa->NumarNod][nod].first)
		{
			Graf[Sursa->NumarNod][nod].second = 0;
			Graf[nod][Sursa->NumarNod].second = Graf[nod][Sursa->NumarNod].second + Graf[Sursa->NumarNod][nod].first;
			Retea[nod]->Exces = Graf[Sursa->NumarNod][nod].first;
			Sursa->Exces = Sursa->Exces - Graf[Sursa->NumarNod][nod].first;
		}
	}
}

void Pompare(std::vector < std::vector < std::pair < int, int > > >& Graf, Nod*& DeTrimis, Nod*& DePrimit)
{
	int flux = (DeTrimis->Exces < Graf[DeTrimis->NumarNod][DePrimit->NumarNod].second) ? DeTrimis->Exces : Graf[DeTrimis->NumarNod][DePrimit->NumarNod].second;
	Graf[DeTrimis->NumarNod][DePrimit->NumarNod].second = Graf[DeTrimis->NumarNod][DePrimit->NumarNod].second - flux;
	Graf[DePrimit->NumarNod][DeTrimis->NumarNod].second = Graf[DePrimit->NumarNod][DeTrimis->NumarNod].second + flux;
	DeTrimis->Exces = DeTrimis->Exces - flux;
	DePrimit->Exces = DePrimit->Exces + flux;
}

void Descarca(std::vector < std::vector < std::pair < int, int > > >& Graf, std::vector < Nod* >& Retea, Nod*& DeDescarcat)
{
	Nod* vecin;

	while (DeDescarcat->Exces > 0)
	{
		if (DeDescarcat->Curent == DeDescarcat->Vecini.end())
		{
			Inaltare(Graf, Retea, DeDescarcat);
			DeDescarcat->Curent = DeDescarcat->Vecini.begin();
		}
		else
		{
			vecin = Retea[*DeDescarcat->Curent];
			if (Graf[DeDescarcat->NumarNod][vecin->NumarNod].second > 0 && DeDescarcat->Inaltime == vecin->Inaltime + 1)
			{
				Pompare(Graf, DeDescarcat, vecin);
			}
			else
			{
				DeDescarcat->Curent++;
			}
		}
	}
}

void Inaltare(std::vector < std::vector < std::pair < int, int > > >& Graf, std::vector < Nod* >& Retea, Nod*& DeInaltat)
{
	int nrNoduri = (int)Graf.size();
	int inaltimeMinima = 10 * nrNoduri;
	
	for (int vecin = 0; vecin < nrNoduri; vecin = vecin + 1)
	{
		if (Retea[vecin]->Inaltime < inaltimeMinima && Graf[DeInaltat->NumarNod][vecin].second > 0)
			inaltimeMinima = Retea[vecin]->Inaltime;
	}
	DeInaltat->Inaltime = 1 + inaltimeMinima;
}

int PompareTopologica(std::vector < std::vector < std::pair < int, int > > > Graf, int Sursa, int Dest)
{
	int nrNoduri = (int)Graf.size(), inaltimeVeche;
	Nod* deAdaugat;
	std::vector < Nod* > retea;
	std::list < Nod* > lista;
	std::list < Nod* >::iterator nod;

	for (int i = 0; i < nrNoduri; ++i)
	{
		retea.push_back(new Nod(i));
		for (int j = 0; j < nrNoduri; ++j)
		{
			if (Graf[i][j].first || Graf[j][i].first)
				retea[i]->Vecini.push_back(j);
		}		
		
		retea[i]->Curent = retea[i]->Vecini.begin();
		if (i == Sursa || i == Dest)
			continue;
		
		lista.push_back(retea[i]);
	}
	InitializarePreflux(Graf, retea, retea[Sursa]);
	nod = lista.begin();

	while (nod != lista.end())
	{
		inaltimeVeche = (*nod)->Inaltime;
		Descarca(Graf, retea, *nod);
		if ((*nod)->Inaltime > inaltimeVeche)
		{
			deAdaugat = *nod;
			lista.erase(nod);
			lista.push_front(deAdaugat);
			nod = lista.begin();
		}
		nod++;
	}

	return retea[Dest]->Exces;
}

void Problema1(std::string InFile, std::string OutFile)
{
	int nrNoduri, nrMuchii, sursa, dest, cost;
	std::vector < std::vector < std::pair < int, int > > > graf;
	
	std::ifstream in(InFile);

	in >> nrNoduri >> nrMuchii;

	graf = std::vector < std::vector < std::pair < int, int > > >(nrNoduri);

	for (int muchie = 0; muchie < nrMuchii; muchie = muchie + 1)
	{
		in >> sursa >> dest >> cost;
		graf[sursa].push_back({ dest, cost });
	}

	in.close();

	std::ofstream out(OutFile);

	out << EdmondsKarp(graf, 0, nrNoduri - 1);

	out.close();
}

void Problema2(std::string InFile, std::string OutFile)
{
	int nrNoduri, nrMuchii, sursa, dest, cost;
	std::vector < std::vector < std::pair < int, int > > > graf;

	std::ifstream in(InFile);

	in >> nrNoduri >> nrMuchii;

	graf = std::vector < std::vector < std::pair < int, int > > >(nrNoduri, std::vector < std::pair < int, int > >(nrNoduri, {0, 0}));

	for (int muchie = 0; muchie < nrMuchii; muchie = muchie + 1)
	{
		in >> sursa >> dest >> cost;
		graf[sursa][dest] = { cost, cost };
	}

	in.close();

	std::ofstream out(OutFile);

	out << PompareTopologica(graf, 0, nrNoduri - 1);

	out.close();
}

void Problema3(std::string InFile, std::string OutFile)
{
	int nrNoduri, nrMuchii, sursa, dest;

	std::ifstream in(InFile);

	in >> nrNoduri >> nrMuchii;

	G = std::vector < std::vector < int > >(nrNoduri);

	for (int muchie = 0; muchie < nrMuchii; muchie = muchie + 1)
	{
		in >> sursa >> dest;
		Muchii.push_back({ sursa, dest });
		Elim.push_back(false);
		G[sursa].push_back((int)Muchii.size() - 1);
		G[dest].push_back((int)Muchii.size() - 1);
	}

	in.close();

	std::ofstream out(OutFile);

	Euler();
	Ciclu.pop_back();
	for (int& nod : Ciclu)
	{
		out << nod << ' ';
	}
	Ciclu.clear();
	Muchii.clear();
	Elim.clear();
	G.clear();
	out.close();
}

void Test(int NrProblema, int NrTeste)
{
	std::string inFile;
	std::string outFile;
	std::string outData, auxData;

	std::cout << "------Teste problema " << NrProblema << " ------ \n\n";

	for (int i = 1; i <= NrTeste; i = i + 1)
	{
		std::cout << i << " ";

		inFile = std::to_string(NrProblema) + "/" + std::to_string(i) + "-in.txt";
		outFile = std::to_string(NrProblema) + "/" + std::to_string(i) + "-out.txt";

		switch (NrProblema)
		{
		case 1:
			Problema1(inFile, "1/aux-out.txt");
			break;
		case 2:
			Problema2(inFile, "2/aux-out.txt");
			break;
		case 3:
			Problema3(inFile, "3/aux-out.txt");
			break;
		default:
			break;
		}

		std::ifstream toCheck(outFile);
		std::ifstream auxFile(std::to_string(NrProblema) + "/aux-out.txt");


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
}
