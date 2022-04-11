#ifndef _VALIDATOR_H_
#define _VALIDATOR_H_

#include "data.h"

/// <summary>
///		Verifica daca un tip este valid sau nu
/// </summary>
/// <param name="Type">OfferType - tipul de verificat</param>
/// <returns>
///		1 - daca tipul este 1/2/3
///		0 - altfel
/// </returns>
static BOOL ValidateOfferType(OfferType Type);

/// <summary>
///		Verifica daca o data este valida sau nu
/// </summary>
/// <param name="Date">struct tm - data de verificat</param>
/// <returns>
///		1 - daca data este corecta
///		0 - altfel
/// </returns>
static BOOL ValidateOfferDate(struct tm Date);

/// <summary>
///		Verifica daca pretul este valid sau nu
/// </summary>
/// <param name="Price">int - pretul de verificat</param>
/// <returns>
///		1 - daca pretul e numar pozitiv
///		0 - altfel
///	</returns>
static BOOL ValidateOfferPrice(int Price);

/// <summary>
///		Verifica daca o oferta este valida. O oferta este valida daca
///  tipul, data si pretul sunt valide.
/// </summary>
/// <param name="Offer">OFFER - oferta de verificat</param>
/// <param name="Error">char* - sir in care se vor pune erorile</param>
/// <returns>
///		1 - Daca oferta este valida
///		0 - altfel
/// </returns>
BOOL ValidateOffer(OFFER Offer, char** Error);

#endif // !_VALIDATOR_H_
