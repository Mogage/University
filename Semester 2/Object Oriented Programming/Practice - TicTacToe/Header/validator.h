#pragma once

#include "domain.h"

class Validator
{
public:
	bool validateDim(int Dimensiune);

	bool validateTable(string TablaSir, int Dimensiune);

	void validateGame(const Game& GameToVerify);
};

