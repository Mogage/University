#pragma once

#include <string>
#include <vector>
#include <ostream>

using std::string;
using std::vector;

class Game
{
private:
	char Jucator;
	int Id, Dim;
	string Stare;
	vector < vector < char > > Tabla;
public:
	Game();
	Game(int _Id, int _Dim, string _Stare, char _Jucator, string _Tabla);

	char curent() const;
	int id() const;
	int dim() const;
	string stare() const;
	string tablaSir() const;
	vector < vector < char > > tabla() const;
	char cell(int Row, int Col);

	char move(int Row, int Col);

	friend std::ostream& operator<<(std::ostream& os, const Game& _Game);
};

