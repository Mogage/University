#ifndef _SERVICE_H_
#define _SERVICE_H_

#include "repository.h"
#include "validator.h"

typedef struct _SERVICE* SERVICE;

/// <summary>
///		Creeaza un service de oferte
/// </summary>
/// <param name="Service">SERVICE - service de oferte de creat</param>
/// <param name="Repo">REPOSITORY - Repo cu oferte</param>
void CreateService(SERVICE* Service, REPOSITORY Repo);

/// <summary>
///		Distruge un service de oferte
/// </summary>
/// <param name="Service">SERVICE - service de oferte de distrus</param>
void DestroyService(SERVICE* Service);

/// <summary>
///		Creeaza o oferta, o verifica iar daca este valida o adauga in repo
/// </summary>
/// <param name="Service">SERVICE - service de oferte</param>
/// <param name="Type">OfferType - tipul ofertei de adaugat</param>
/// <param name="Destination">char* - destinatia ofertei de adaugat</param>
/// <param name="Date">struct tm - data ofertei de adaugat</param>
/// <param name="Price">int - pretul ofertei de adaugat</param>
/// <returns>
/// NULL - daca nu exista nicio eroare
/// char* - sirul cu erori de afisat
/// </returns>
char* AddOfferService(SERVICE Service, OfferType Type, const char* Destination, struct tm Date, int Price);

/// <summary>
///		Actualizeaza o oferta care exista
/// </summary>
/// <param name="Service">SERVICE - service de oferte</param>
/// <param name="OldType">OfferType - tipul ofertei de actualizat</param>
/// <param name="OldDestination">char* - destinatia ofertei de actualizat</param>
/// <param name="OldDate">struct tm - data ofertei de actualizat</param>
/// <param name="NewType">OfferType - tipul cu care se actualizeaza</param>
/// <param name="NewDestination">char* - destinatia cu care se actualizeaza</param>
/// <param name="NewDate">struct tm - data cu care se actualizeaza</param>
/// <param name="NewPrice">int - pretul cu care se actualizeaza</param>
/// <returns>
/// NULL - daca nu exista nicio eroare
/// char* - sirul cu erori de afisat
/// </returns>
char* UpdateOfferService(SERVICE Service, OfferType OldType, const char* OldDestination, struct tm OldDate,
										  OfferType NewType, const char* NewDestination, struct tm NewDate, int NewPrice);

/// <summary>
///		Sterge o oferta existenta cu datele specificate
/// </summary>
/// <param name="Service">SERVICE - service de oferte</param>
/// <param name="Type">OfferType - tipul ofertei de sters</param>
/// <param name="Destination">char* - destinatia ofertei de sters</param>
/// <param name="Date">struct tm - data ofertei de sters</param>
/// <returns>
/// NULL - daca nu exista nicio eroare
/// char* - sirul cu erori de afisat
/// </returns>
char* DeleteOfferService(SERVICE Service, OfferType Type, const char* Destination, struct tm Date);

/// <summary>
///		Verifica daca o oferta creata cu datele specificate exista in repo
/// </summary>
/// <param name="Service">SERVICE - service de oferte</param>
/// <param name="Type">OfferType - tipul ofertei de cautat</param>
/// <param name="Destination">char* - destinatia ofertei de cautat</param>
/// <param name="Date">struct tm - data ofertei de cautat</param>
/// <returns>
///		1 - daca exista aceasta oferta in repo
///		0 - altfel
/// </returns>
BOOL CheckOfferExistsService(SERVICE Service, OfferType Type, const char* Destination, struct tm Date, char** Errors);

/// <summary>
///		Determina numarul de elemente din repo
/// </summary>
/// <param name="Service">SERVICE - service de oferte</param>
/// <returns> int - numarul de oferte din repo </returns>
int GetCountRepoService(SERVICE Service);

/// <summary>
///		Determina toate ofertele din repo
/// </summary>
/// <param name="Service">SERVICE - service de oferte</param>
/// <param name="allOffers">OFFER* - vector cu toate ofertele din repo</param>
/// <param name="Count">int - numarul de oferte din repo</param>
void GetAllOffersService(SERVICE Service, OFFER** AllOffers, int* Count);

/// <summary>
///		Interschimba valorile a 2 oferte
/// </summary>
/// <param name="">OFFER - prima oferta</param>
/// <param name="">OFFER - a doua oferta</param>
static void Swap(OFFER*, OFFER*);

/// <summary>
///		Functie de comparare in functie de tipul sortarii (crescator / descrescator)
/// </summary>
/// <param name="SortType">int - Tipul de sortare, 1 - crescator, altfel descrescator</param>
/// <param name="ToCheck">int - Valoare de comparat, reprezinta diferenta dintre 2 elemente</param>
/// <returns>
///		1 - Daca ToCheck se incadreaza in intervalul definit de tipul de sortare
///		0 - altfel
/// </returns>
static BOOL CompareOrder(int SortType, int ToCheck);

/// <summary>
///		Sorteaza dupa pret un vector de oferte
/// </summary>
/// <param name="Offers">OFFER* - vectorul de oferte care trebuie sortat</param>
/// <param name="Count">int - dimensiunea vectorului</param>
/// <param name="SortType">int - tipul sortarii, 1 - crescator, altfel descrescator</param>
static void SortAfterPrice(OFFER* Offers, int Count, int SortType);

/// <summary>
///		Sorteaza dupa destinatie lexicografic un vector de oferte
/// </summary>
/// <param name="Offers">OFFER* - vectorul de oferte care trebuie sortat</param>
/// <param name="Count">int - dimensiunea vectorului</param>
/// <param name="SortType">int - tipul sortarii, 1 - crescator, altfel descrescator</param>
static void SortAfterDestination(OFFER* Offers, int Count, int SortType);

/// <summary>
///		Determina in Offers ofertele sortate dupa:
///		Daca FilterOrder = 1, dupa pret, altfel dupa destinatie
///		Daca TypeOrder = 1, crescator, altfel descrescator
/// </summary>
/// <param name="Service"><SERVICE - service de oferte/param>
/// <param name="FilterOrder">int - criteriul dupa care sa se sorteze</param>
/// <param name="TypeOrder">int - modul in care sa se sorteze</param>
/// <param name="Offers">OFFER* - vector cu toate ofertele din repo sortate dupa un criteriu</param>
/// <param name="Count">int - numarul de oferte din repo</param>
void GetOffersOrderedService(SERVICE Service, int FilterOrder, int TypeOrder, OFFER** Offers, int* Count);

/// <summary>
///		Compara in funtie de FilterType un atribut al unei oferte cu filtrul specificat
/// </summary>
/// <param name="FilterType">int - tipul de filtrare, 1-destinatie, 2-tip, 3-pret</param>
/// <param name="Offer">OFFER - oferta pentru care se verifica un atribut</param>
/// <param name="Filter">void* - filtrul cu care se compara un atribut al ofertei</param>
/// <returns>
///		1 - daca atributul specificat in functie de FilterType este egal cu Filter
///		0 - altfel
/// </returns>
static BOOL CompareFilter(int FilterType, OFFER Offer, const void* Filter);

/// <summary>
///		Determina in Offers toate ofertele filtrate dupa Filter
/// </summary>
/// <param name="Service">SERVICE - serviciu de oferte</param>
/// <param name="FilterType">int - tipul de filtrare, 1-destinatie, 2-tip, 3-pret</param>
/// <param name="Filter">void* - filtrul cu care se compara un atribut al ofertei</param>
/// <param name="Offers">OFFER* - vector cu toate ofertele din repo filtrate dupa un filtru</param>
/// <param name="Count">int - numarul de oferte filtrare</param>
void GetOffersFilteredService(SERVICE Service, int FilterType, const void* Filter, OFFER** Offers, int* Count);

#endif // !_SERVICE_H
