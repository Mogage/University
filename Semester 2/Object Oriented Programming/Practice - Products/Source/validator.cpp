#include "validator.h"

void Validator::ValidateProduct(Produs& ToVerify)
{
	string errors;

	if ("" == ToVerify.name())
		errors = errors + "Nume produs invalid.\n";
	if (ToVerify.price() < 1 || ToVerify.price() > 100)
		errors = errors + "Pret produs invalid.\n";

	if (errors != "")
		throw ValidationError(errors);
}
