#include "service.h"

void Service::createGame(int Dim, string TablaSir, char Jucator)
{
    Game created{ Repo.size() + 1, Dim, "Neinceput", Jucator, TablaSir};
    Valid.validateGame(created);
    Repo.addGame(created);
}

char Service::move(int Id, int Row, int Col)
{
    Game game = Repo.findId(Id);
    if (game.stare() == "Terminat" || game.tabla()[Row][Col] != '-')
        return game.tabla()[Row][Col];
    return Repo.move(Id, Row, Col);
}

Game Service::find(int Id)
{
    return Repo.findId(Id);
}

vector<Game>& Service::all()
{
    return Repo.getAll();
}

