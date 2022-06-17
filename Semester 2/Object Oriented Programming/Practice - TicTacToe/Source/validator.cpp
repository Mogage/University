#include "validator.h"
#include "exceptions.h"

bool Validator::validateDim(int Dimensiune)
{
    if (Dimensiune < 3 || Dimensiune > 5)
        return false;

    return true;
}

bool Validator::validateTable(string TablaSir, int Dimensiune)
{
    if (TablaSir.length() != Dimensiune * Dimensiune)
        return false;

    for (char& cell : TablaSir)
    {
        if (cell != 'X' && cell != 'O' && cell != '-')
            return false;
    }

    return true;
}

void Validator::validateGame(const Game& GameToVerify)
{
    string errors;
    if (!validateDim(GameToVerify.dim()))
        errors = errors + "Dimensiune joc invalida.\n";
    if (!validateTable(GameToVerify.tablaSir(), GameToVerify.dim()))
        errors = errors + "Tabla de joc invalida.\n";
    char jucator = GameToVerify.curent();
    if (jucator != 'X' && jucator != 'O')
        errors = errors + "Jucator invalid.\n";

    if (0 != errors.size())
    {
        throw ValidationError(errors);
    }
}
