#include <stdio.h>
#include <stdlib.h>

void CalcDivisors(int Number, int* TotalZero, int* TwoDivisor, int* FiveDivisor)
{
	while (Number % 10 == 0)
	{
		*TotalZero = *TotalZero + 1;
		Number = Number / 10;
	}
	while (Number % 2 == 0)
	{
		Number = Number / 2;
		*TwoDivisor = *TwoDivisor + 1;
	}
	while (Number % 5 == 0)
	{
		Number = Number / 5;
		*FiveDivisor = *FiveDivisor + 1;
	}
}

int Minim(int Number1, int Number2)
{
	return (Number1 > Number2) ? Number2 : Number1;
}

int CalcTotalZero()
{
	int number = -1;
	int totalZero = 0, twoDivisor = 0, fiveDivisor = 0;
	while (number != 0)
	{
		scanf_s("%d", &number);
		if (number != 0)
		{
			CalcDivisors(number, &totalZero, &twoDivisor, &fiveDivisor);
		}
	}
	totalZero = totalZero + Minim(twoDivisor, fiveDivisor);
	return totalZero;
}

void Problema9()
{
	int totalZero;
	totalZero = CalcTotalZero();
	printf("Numarul cifrelor de 0 de la finalul produsului numerelor este: %d", totalZero);
}

double CalcXLaN(double x, int n)
{
	double rezultat = x;
	n = n - 1;
	while (n)
	{
		if (n % 2 == 1)
		{
			rezultat = rezultat * x;
		}
		x = x * x;
		n = n / 2;
	}
	return rezultat;
}

void Problema12()
{
	int n;
	double rezultat, x;
	scanf_s("%lf %d", &x, &n);
	rezultat = CalcXLaN(x, n);
	printf("x^n este: %lf\n", rezultat);
}

int main()
{
	/*
		Citeste un sir de numere naturale nenule terminat cu 0 si determina
    numarul cifrelor 0 in care se termina numarul produs al numerelor citite.
	*/
	//Problema9();

	/*
		Determina valoarea x^n (x este un numar real dat, n este un numar natural dat), 
	utilizand operatii de inmultire si de ridicare la patrat.
	*/
	Problema12();
	return 0;
}