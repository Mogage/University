#include "domain.h"

Game::Game()
{
	Id = -1;
	Dim = -1;
	Jucator = '\0';
	Stare = "";
}

Game::Game(int _Id, int _Dim, string _Stare, char _Jucator, string _Tabla)
{
	Id = _Id;
	Dim = _Dim;
	Jucator = _Jucator;
	Stare = _Stare;
	if (_Dim * _Dim != _Tabla.length())
		return;
	int row = 0, col = 0;
	Tabla = vector < vector < char > >(Dim, vector < char >(Dim));
	for (auto& cell : _Tabla)
	{
		Tabla[row][col] = cell;
		col = col + 1;
		if (col == Dim)
		{
			col = 0;
			row = row + 1;
		}
	}
}

char Game::curent() const
{
	return Jucator;
}

int Game::id() const
{
	return Id;
}

int Game::dim() const
{
	return Dim;
}

string Game::stare() const
{
	return Stare;
}

string Game::tablaSir() const
{
	string stringTable;
	for (const auto& linie : Tabla)
	{
		for (const char& cell : linie)
		{
			stringTable.push_back(cell);
		}
	}
	return stringTable;
}

vector<vector<char>> Game::tabla() const
{
	return Tabla;
}

char Game::cell(int Row, int Col)
{
	return Tabla[Row][Col];
}

bool Game::checkWin(const vector < vector < char > >& _Tabla)
{
	bool toRet = false;
	int nr = Tabla.size();

	for (int cont = 0; cont < nr; ++cont)
	{
		toRet = true;
		for (int cont2 = 1; cont2 < nr; ++cont2)
		{
			if (Tabla[cont][0] == '-' || Tabla[cont][0] != Tabla[cont][cont2])
			{
				toRet = false;
				break;
			}
		}
		if (toRet) return toRet;
	}

	for (int cont = 0; cont < nr; ++cont)
	{
		toRet = true;
		for (int cont2 = 1; cont2 < nr; ++cont2)
		{
			if (Tabla[0][cont] == '-' || Tabla[0][cont] != Tabla[cont2][cont])
			{
				toRet = false;
				break;
			}
		}
		if (toRet) return toRet;
	}

	toRet = true;
	for (int cont = 1; cont < nr; ++cont)
	{
		if (Tabla[0][0] == '-' || Tabla[0][0] != Tabla[cont][cont])
		{
			toRet = false;
			break;
		}
	}
	if (toRet) return toRet;

	toRet = true;
	for (int cont = 1; cont < nr; ++cont)
	{
		if (Tabla[0][nr - 1] == '-' || Tabla[0][nr - 1] != Tabla[cont][nr - 1 - cont])
		{
			toRet = false;
			break;
		}
	}
	if (toRet) return toRet;

	return toRet;
}

char Game::move(int Row, int Col)
{
	if (Stare == "Terminat")
		return Jucator;
	char toReturn = Jucator;
	Tabla[Row][Col] = Jucator;
	if ('X' == Jucator)
		Jucator = 'O';
	else Jucator = 'X';
	int nrLin = 0;
	for (auto& linie : Tabla)
	{
		for (char& cell : linie)
		{
			if ('-' == cell)
			{
				nrLin++;
			}
		}
	}

	if (checkWin(Tabla) || 0 == nrLin)
	{
		Stare = "Terminat";
	}
	else if (nrLin > 0 && nrLin < Dim * Dim)
	{
		Stare = "In derulare";
	}
	return toReturn;
}

std::ostream& operator<<(std::ostream& os, const Game& _Game)
{
	os << _Game.id() << ' '
		<< _Game.dim() << ' '
		<< _Game.tablaSir() << ' ' 
		<< _Game.curent() << ' ' 
		<< _Game.stare() << '\n';

	return os;
}
