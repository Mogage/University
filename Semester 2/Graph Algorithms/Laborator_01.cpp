/*
1.  Fie un fisier ce contine un graf neorientat reprezentat sub forma:
prima linie contine numarul nodurilor si al muchilor iar urmatoarele randuri muchiile grafului. 
	Sa se scrie un program in C/C++ care sa citeasca fisierul si sa reprezinte/stocheze un graf folosind 
matricea de adiacenta, lista de adiacenta si matricea de incidenta. Sa se converteasca un graf dintr-o forma de reprezentare in alta.

Fisier -> matrice de adiacenta -> lista adiacenta -> matrice de incidenta -> lista adiacenta -> matrice de adiacenta -> lista.

exemplu fisier
in.txt
4
1 2
3 4
2 3
4 2

2. Fie un graf reprezentat sub o anumita forma (graful este citit dintr-un fisier). Sa se rezolve:
a. sa se determine nodurile izolate dintr-un graf.
b. sa se determine daca graful este regular.
c. pentru un graf reprezentat cu matricea de adiacenta sa se determine matricea distantelor.
d. pentru un graf reprezentat cu o matrice de adiacenta sa se determine daca este conex.
*/

#include <iostream>
#include <fstream>
#include <vector>
std::ifstream fin("in.txt");

#define INF 10000

typedef struct _VECINI
{
	int Value;
	_VECINI* Next;
} VECINI;

void AfisareMatrice(int Linii, int Coloane, std::vector< std::vector < bool > > Matrice)
{
	for (int i = 1; i <= Linii; i = i + 1)
	{
		for (int j = 1; j <= Coloane; j = j + 1)
		{
			std::cout << Matrice[i][j] << " ";
		}
		std::cout << "\n";
	}
	std::cout << "\n";
}

void MatriceToLista(int Varfuri, std::vector< std::vector < bool > > Matrice, std::vector < VECINI* >& Lista)
{
	VECINI* vecinNou;
	for (int i = Varfuri; i >= 1; i = i - 1)
	{
		for (int j = Varfuri; j >= 1; j = j - 1)
		{
			if (true == Matrice[i][j])
			{
				vecinNou = new VECINI;
				vecinNou->Value = j;
				vecinNou->Next = Lista[i];
				Lista[i] = vecinNou;
			}
		}
	}
}

void Problema1()
{
	int nrVarfuri, nrMuchii;
	int varf1, varf2;
	int nrMuchiiCurent = 1;
	VECINI* vecinCurent, * deEliminat;
	fin >> nrVarfuri >> nrMuchii;
	std::vector < std::vector < bool > > matriceAdiacenta(nrVarfuri + 1, std::vector< bool >(nrVarfuri + 1, false));
	std::vector < std::vector < bool > > matriceIncidenta(nrVarfuri + 1, std::vector< bool >(nrMuchii + 1, false));
	std::vector < VECINI* > listaAdiacenta(nrVarfuri + 1, NULL);
	for (int i = 0; i < nrMuchii; i = i + 1)
	{
		fin >> varf1 >> varf2;
		matriceAdiacenta[varf1][varf2] = 1;
		matriceAdiacenta[varf2][varf1] = 1;
	}
	AfisareMatrice(nrVarfuri, nrVarfuri, matriceAdiacenta);

	MatriceToLista(nrVarfuri, matriceAdiacenta, listaAdiacenta);

	for (int i = 1; i <= nrVarfuri; i = i + 1)
	{
		std::cout << i << ": ";
		vecinCurent = listaAdiacenta[i];
		while (vecinCurent != NULL)
		{
			if (matriceIncidenta[i][nrMuchiiCurent - 1] == 0)
			{
				matriceIncidenta[i][nrMuchiiCurent] = true;
				matriceIncidenta[vecinCurent->Value][nrMuchiiCurent] = true;
				nrMuchiiCurent = nrMuchiiCurent + 1;
			}
			std::cout << vecinCurent->Value << " ";
			deEliminat = vecinCurent;
			vecinCurent = vecinCurent->Next;
			delete deEliminat;
			deEliminat = NULL;
		}
		std::cout << "\n";
	}
	std::cout << "\n";

	for (int i = 1; i <= nrVarfuri; i = i + 1)
	{
		while (listaAdiacenta[i] != NULL)
		{
			std::cout << "nesters";
			listaAdiacenta[i] = listaAdiacenta[i]->Next;
		}
	}
	
	AfisareMatrice(nrVarfuri, nrMuchii, matriceIncidenta);
}

void Problema2()
{
	bool izolat, existaIzolat, regular, conex;
	int nrVarfuri, nrMuchii;
	int varf1, varf2;
	int grad1 = 0, grad2 = 0;
	fin >> nrVarfuri >> nrMuchii;
	std::vector < std::vector < int > > matriceAdiacenta(nrVarfuri + 1, std::vector< int >(nrVarfuri + 1, 0));

	for (int i = 0; i < nrMuchii; i = i + 1)
	{
		fin >> varf1 >> varf2;
		matriceAdiacenta[varf1][varf2] = true;
		matriceAdiacenta[varf2][varf1] = true;
	}
	existaIzolat = false;
	for (int i = 1; i <= nrVarfuri; i = i + 1)
	{
		izolat = true;
		for (int j = 1; j <= nrVarfuri; j = j + 1)
		{
			if (1 == matriceAdiacenta[i][j])
			{
				izolat = false;
			}
		}
		if (true == izolat)
		{
			std::cout << i << " ";
			existaIzolat = true;
		}
	}
	if (false == existaIzolat)
	{
		std::cout << "Nu exista noduri izolate";
	}
	std::cout << "\n";
	for (int i = 1; i <= nrVarfuri; ++i)
	{
		grad1 = grad1 + matriceAdiacenta[1][i];
	}
	regular = true;
	for (int i = 2; i <= nrVarfuri; i = i + 1)
	{
		grad2 = 0;
		for (int j = 1; j <= nrVarfuri; j = j + 1)
		{
			grad2 = grad2 + matriceAdiacenta[i][j];
		}
		if (grad1 != grad2)
		{
			std::cout << "Graful nu este regular\n";
			regular = false;
			break;
		}
		grad1 = grad2;
	}
	if (true == regular)
	{
		std::cout << "Graful este regular\n";
	}

	for (int i = 1; i <= nrVarfuri; i = i + 1)
	{
		for (int j = 1; j <= nrVarfuri; j = j + 1)
		{
			if (i == j)
			{
				matriceAdiacenta[i][j] = 0;
			}
			else if (0 == matriceAdiacenta[i][j])
			{
				matriceAdiacenta[i][j] = INF;
			}
		}
	}

	for (int k = 1; k <= nrVarfuri; k = k + 1)
	{
		for (int i = 1; i <= nrVarfuri; i = i + 1)
		{
			for (int j = 1; j <= nrVarfuri; j = j + 1)
			{
				if (matriceAdiacenta[i][j] > matriceAdiacenta[i][k] + matriceAdiacenta[k][j])
				{
					matriceAdiacenta[i][j] = matriceAdiacenta[i][k] + matriceAdiacenta[k][j];
				}
			}
		}
	}

	conex = true;
	for (int i = 1; i <= nrVarfuri; i = i + 1)
	{
		for (int j = 1; j <= nrVarfuri; j = j + 1)
		{
			if (INF == matriceAdiacenta[i][j])
			{
				std::cout << "-1 ";
				conex = false;
			}
			else
			{
				std::cout << matriceAdiacenta[i][j] << " ";
			}
		}
		std::cout << "\n";
	}

	if (true == conex)
	{
		std::cout << "Graful este conex\n";
	}
	else
	{
		std::cout << "Graful nu este conex\n";
	}
}

int main()
{
	Problema1();
	
	//Problema2();

	fin.close();
	return 0;
}