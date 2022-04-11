#ifndef _USERINTERFACE_H_
#define _USERINTERFACE_H_

#include "service.h"

typedef struct _USERINT* USERINT;

/// <summary>
///		Creeaza tipul interfata utilizator
/// </summary>
/// <param name="UserInt">USERINT - tip interfata utilizator de creat</param>
/// <param name="Service">SERVICE - service de oferte</param>
void CreateUserInt(USERINT* UserInt, SERVICE Service);

/// <summary>
///		Distruge tipul interfata utilizator creata
/// </summary>
/// <param name="UserInt">USERINT - tip interfata utilizator de distrus</param>
void DestroyUserInt(USERINT* UserInt);

/// <summary>
///		Citeste de la utilizator datele unei oferte si o adauga in repo
/// </summary>
/// <param name="UserInt">USERINT - tip interfata utilizator</param>
static void AddOfferUserInt(USERINT UserInt);

/// <summary>
///		Citeste datele ofertei care trebuie actualizate
/// </summary>
/// <param name="Type">OfferType - tipul ofertei de actualizat</param>
/// <param name="Destination">char* - destinatia ofertei de actualizat</param>
/// <param name="Date"><struct tm - data ofertei de actualizat/param>
static void ReadOldOfferUpdateUserInt(OfferType* Type, char** Destination, struct tm* Date);

/// <summary>
///		Citeste noile date cu care sa se actualizeze o oferta
/// </summary>
/// <param name="Type">OfferType - tipul nou cu care se actualizeaza</param>
/// <param name="Destination">char* - destinatia noua</param>
/// <param name="Date">struct tm - data noua</param>
/// <param name="Price">int - pretul nou</param>
static void ReadNewOfferUpdateUserInt(OfferType* Type, char** Destination, struct tm* Date, int* Price);

/// <summary>
///		Actualizeaza o oferta din repo
/// </summary>
/// <param name="UserInt">USERINT - tip interfata utilizator</param>
static void UpdateOfferUserInt(USERINT UserInt);

/// <summary>
///		Sterge o oferta din repo cu datele citite de la utilizator
/// </summary>
/// <param name="UserInt">USERINT - tip interfata utilizator</param>
static void DeleteOfferUserInt(USERINT UserInt);

/// <summary>
///		Afiseaza toate ofertele existente in repo
/// </summary>
/// <param name="UserInt">USERINT - tip interfata utilizator</param>
static void PrintAllOffers(USERINT UserInt);

/// <summary>
///		Afiseaza toate ofertele existente ordonate dupa un criteriu
///     pret, destinatie, crescator / descrescator
/// </summary>
/// <param name="UserInt">USERINT - tip interfata utilizator</param>
static void PrintOrderedOffers(USERINT UserInt);

/// <summary>
///		Afiseaza toate ofertele existente filtrate dupa un criteriu:
///		destinatie, tip, pret
/// </summary>
/// <param name="UserInt">USERINT - tip interfata utilizator</param>
static void PrintFilteredOffers(USERINT UserInt);

/// <summary>
///		Afiseaza meniul principal cu care interactioneaza utilizatorul
/// </summary>
static void PrintMenu();

/// <summary>
///		Functia main de rulare a aplicatiei
/// </summary>
/// <param name="UserInt">USERINT - tip interfata utilizator</param>
void RunApp(USERINT UserInt);

#endif // !_USERINTERFACE_H_
