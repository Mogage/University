#include <iostream>
#include <fstream>
#include <map>
#include <vector>
#include <queue>
#include <string>
#include <assert.h>

std::vector < int > PruferCoding(std::vector < int > Arbore, std::vector < int > Grad, int Radacina);
std::vector < int > PruferDecoding(std::vector < int > Cod);

class HuffmanNode
{
public:
	int Freq;
	char Litera;
	HuffmanNode* Left, * Right;

	HuffmanNode(char _Litera, int _Freq, HuffmanNode* _Left = NULL, HuffmanNode* _Right = NULL) : Freq{ _Freq }, Litera{ _Litera }, Left{ _Left }, Right{ _Right }{}

	friend std::ostream& operator <<(std::ostream& Out, const HuffmanNode* Data)
	{
		Out << Data->Litera << " " << Data->Freq << "\n";
		return Out;
	}

	friend std::ostream& operator <<(std::ostream& Out, const HuffmanNode Data)
	{
		Out << Data.Litera << " " << Data.Freq << "\n";
		return Out;
	}
};
class Compare
{
public:
	bool operator()(HuffmanNode*& First, HuffmanNode*& Second)
	{
		if(First->Freq != Second->Freq)
			return (First->Freq > Second->Freq);

		return (First->Litera > Second->Litera);
	}
};

HuffmanNode* HuffmanCoding(std::map < char, int > Alfabet);
std::string HuffmanDecoding(HuffmanNode* Radacina, std::string Cod);
void GetCodes(HuffmanNode* Nod, std::string Cod, std::map < char, std::string >& Coduri);
void CleanUp(HuffmanNode* Nod);

class Muchie
{
public:
	int Sursa, Destinatie, Cost;
	Muchie(int _Sursa, int _Dest, int _Cost) : Sursa{ _Sursa }, Destinatie{ _Dest }, Cost{ _Cost }{}
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
	friend std::ostream& operator <<(std::ostream& Out, Muchie Data)
	{
		Out << Data.Sursa << "-" << Data.Destinatie << ": " << Data.Cost << "\n";

		return Out;
	}
};
class Set
{
private:
	int Size;
	int* Rank, * Parinte;
public:
	Set(int _Size) :Size{ _Size }
	{
		Rank = new int[Size];
		Parinte = new int[Size];
		for (int count = 0; count < Size; count = count + 1)
		{
			Parinte[count] = count;
			Rank[count] = 0;
		}
	}
	~Set()
	{
		delete[] Rank;
		delete[] Parinte;
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

std::vector<std::pair<int, int>> Kruskal(std::vector<Muchie> Muchii, int NrNoduri, int& TotalSum);

void Problema1(std::string InFile, std::string OutFile);
void Problema2(std::string InFile, std::string OutFile);
void Problema3(std::string InFile, std::string OutFile);
void Problema4(std::string InFile, std::string OutFile);
void Problema5(std::string InFile, std::string OutFile);

void Test(int NrProblema, int NrTeste);

int main(int argc, char* argv[])
{
	Test(1, 9);
	Test(2, 9);
	Test(3, 5);
	Test(4, 5);
	Test(5, 10);

	return 0;
}

std::vector < int > PruferCoding(std::vector < int > Arbore, std::vector < int > Grad, int Radacina)
{
	int nrNoduri = (int)Arbore.size(), frunzaMinima, nod, urmator;
	std::vector < int > prufer;

	for (nod = 0; nod < nrNoduri && Grad[nod] != 1; nod = nod + 1);
	frunzaMinima = nod;

	for (int i = 0; i < nrNoduri - 1; i = i + 1)
	{
		urmator = Arbore[frunzaMinima];
		prufer.push_back(urmator);
		Grad[urmator] = Grad[urmator] - 1;
		if (urmator != Radacina && Grad[urmator] == 1 && urmator < nod)
		{
			frunzaMinima = urmator;
		}
		else
		{
			nod = nod + 1;
			while (nod < nrNoduri && (Grad[nod] != 1 || nod == Radacina)) nod = nod + 1;
			frunzaMinima = nod;
		}
	}

	return prufer;
}

std::vector < int > PruferDecoding(std::vector < int > Cod)
{
	int nrNoduri = (int)Cod.size() + 1, ptr, frunza;
	std::vector < int > arbore(nrNoduri, -1);
	std::vector < int > grad(nrNoduri, 1);

	for (const auto& nod : Cod)
	{
		grad[nod] = grad[nod] + 1;
	}

	for (ptr = 0; ptr < nrNoduri - 1 && grad[ptr] != 1; ptr = ptr + 1);
	frunza = ptr;

	for (const auto& nod : Cod)
	{
		arbore[frunza] = nod;
		grad[nod] = grad[nod] - 1;
		if (grad[nod] == 1 && nod < ptr)
		{
			frunza = nod;
		}
		else
		{
			ptr = ptr + 1;
			while (ptr < nrNoduri && grad[ptr] != 1) ptr = ptr + 1;
			frunza = ptr;
		}
	}

	return arbore;
}

HuffmanNode* HuffmanCoding(std::map < char, int > Alfabet)
{
	std::priority_queue < HuffmanNode* , std::vector < HuffmanNode* > , Compare> coada, toPrint;
	HuffmanNode* left, * right;
	size_t size = Alfabet.size();

	for (const auto& litera : Alfabet)
	{
		coada.push(new HuffmanNode(litera.first, litera.second));
	}

	for(size_t i = 1; i < size; i = i + 1)
	{
		left = coada.top();
		coada.pop();
		right = coada.top();
		coada.pop();

		char smallest = (left->Litera < right->Litera) ? left->Litera : right->Litera;

		coada.push(new HuffmanNode(smallest, left->Freq + right->Freq, left, right));
	}

	return coada.top();
}

std::string HuffmanDecoding(HuffmanNode* Radacina, std::string Cod)
{
	std::string text = "";
	HuffmanNode* curent = Radacina;
	for (const auto& bit : Cod)
	{
		if (curent == NULL)
		{
			curent = Radacina;
			continue;
		}

		if (bit == '0')
		{
			curent = curent->Left;
		}
		else
		{
			curent = curent->Right;
		}
		if (curent && !curent->Left && !curent->Right)
		{
			text = text + curent->Litera;
			curent = Radacina;
		}
	}

	return text;
}

void GetCodes(HuffmanNode* Nod, std::string Cod, std::map<char, std::string>& Coduri)
{
	if (NULL == Nod)
	{
		return;
	}

	if (!Nod->Left && !Nod->Right)
	{
		Coduri[Nod->Litera] = Cod;
	}

	GetCodes(Nod->Left, Cod + "0", Coduri);
	GetCodes(Nod->Right, Cod + "1", Coduri);
}

void CleanUp(HuffmanNode* Nod)
{
	if (NULL != Nod)
	{
		CleanUp(Nod->Left);
		CleanUp(Nod->Right);

		delete Nod;
	}
}

std::vector<std::pair<int,int>> Kruskal(std::vector<Muchie> Muchii, int NrNoduri, int& TotalSum)
{
	std::vector<std::pair<int, int>> arbore;

	Set disjointSet(NrNoduri);

	sort(Muchii.begin(), Muchii.end());

	TotalSum = 0;
	for (const auto& muchie : Muchii)
	{
		if (disjointSet.find(muchie.Sursa) != disjointSet.find(muchie.Destinatie))
		{
			TotalSum = TotalSum + muchie.Cost;
			arbore.push_back(std::make_pair(muchie.Sursa, muchie.Destinatie));
			disjointSet.union_sets(muchie.Sursa, muchie.Destinatie);
		}
	}

	return arbore;
}

void Problema1(std::string InFile, std::string OutFile)
{
	int nrNoduri, nod, radacina = 0;
	std::vector < int > arbore, grad;
	std::ifstream in(InFile);

	in >> nrNoduri;
	grad = std::vector < int >(nrNoduri, 0);
	for (int count = 0; count < nrNoduri; count = count + 1)
	{
		in >> nod;
		if (-1 == nod)
		{
			radacina = count;
		}
		else
		{
			grad[nod] = grad[nod] + 1;
			grad[count] = grad[count] + 1;
		}
		arbore.push_back(nod);
	}

	in.close();

	std::ofstream out(OutFile);

	std::vector < int > prufer = PruferCoding(arbore, grad, radacina);

	out << prufer.size() << "\n";
	for (const auto& pruferCode : prufer)
	{
		out << pruferCode << " ";
	}

	out.close();
}

void Problema2(std::string InFile, std::string OutFile)
{
	int nrValori, valoare;
	std::vector < int > cod, arbore;

	std::ifstream in(InFile);

	in >> nrValori;
	for (int count = 0; count < nrValori; count = count + 1)
	{
		in >> valoare;
		cod.push_back(valoare);
	}

	in.close();

	arbore = PruferDecoding(cod);
	
	std::ofstream out(OutFile);

	out << arbore.size() << "\n";
	for (const auto& nod : arbore)
	{
		out << nod << " ";
	}

	out.close();
}

void Problema3(std::string InFile, std::string OutFile)
{
	std::string input;
	std::ifstream in(InFile);
	std::map < char, int > alfabet;
	std::map < char, std::string > coduri;

	std::getline(in, input);

	in.close();

	for (const auto& caracter : input)
	{
		alfabet[caracter]++;
	}

	HuffmanNode* top = HuffmanCoding(alfabet);
	GetCodes(top, "", coduri);
	CleanUp(top);

	std::ofstream out(OutFile);

	out << alfabet.size() << "\n";

	for (const auto& litera : alfabet)
	{
		out << litera.first << " " << litera.second << "\n";
	}

	for (const auto& caracter : input)
	{
		out << coduri[caracter];
	}

	out.close();
}

void Problema4(std::string InFile, std::string OutFile)
{
	int nrCaractere, freq;
	char litera;
	std::map < char, int > alfabet;
	std::string cod;
	std::ifstream in(InFile);

	in >> nrCaractere;
	for (int count = 0; count < nrCaractere; count = count + 1)
	{
		in.get();
		in.get(litera);
		in >> freq;
		alfabet[litera] = freq;
	}
	in >> cod;

	in.close();

	HuffmanNode* radacina = HuffmanCoding(alfabet);
	
	std::ofstream out(OutFile);

	out << HuffmanDecoding(radacina, cod);

	out.close();
}

void Problema5(std::string InFile, std::string OutFile)
{
	int nrNoduri, nrMuchii, sursa, destinatie, cost, costTotal;
	std::vector < Muchie > muchii;
	std::vector < std::pair < int, int > > arbore;

	std::ifstream in(InFile);

	in >> nrNoduri >> nrMuchii;

	for (int count = 0; count < nrMuchii; count = count + 1)
	{
		in >> sursa >> destinatie >> cost;
		muchii.push_back(Muchie(sursa, destinatie, cost));
	}

	in.close();

	arbore = Kruskal(muchii, nrNoduri, costTotal);

	std::ofstream out(OutFile);

	out << costTotal << "\n" << arbore.size() << "\n";

	std::sort(arbore.begin(), arbore.end());

	for (const auto& muchie : arbore)
	{
		out << muchie.first << " " << muchie.second << "\n";
	}

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
		case 4:
			Problema4(inFile, "4/aux-out.txt");
			break;
		case 5:
			Problema5(inFile, "5/aux-out.txt");
			break;
		default:
			break;
		}

		std::ifstream toCheck(outFile);
		std::ifstream auxFile(std::to_string(NrProblema) + "/aux-out.txt");

		switch (NrProblema)
		{
		case 5:
			std::getline(toCheck, outData);
			std::getline(auxFile, auxData);
			assert(outData == auxData);
			std::getline(toCheck, outData);
			std::getline(auxFile, auxData);
			assert(outData == auxData);
			break;
		default:
			while (std::getline(toCheck, outData))
			{
				std::getline(auxFile, auxData);
				outData.erase(std::remove_if(outData.begin(), outData.end(), isspace), outData.end());
				auxData.erase(std::remove_if(auxData.begin(), auxData.end(), isspace), auxData.end());

				assert(outData == auxData);
			}
			break;
		}

		auxFile.close();
		toCheck.close();
	}

	std::cout << "\t" << NrTeste << "/" << NrTeste << "\n";
}
