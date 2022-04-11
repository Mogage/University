#ifndef _REPOSITORY_H_
#define _REPOSITORY_H_

#include "data.h"

#define INITIAL_SIZE		1048576
#define STATUS_EXISTENT		-2
#define STATUS_INEXISTENT	-3

typedef struct _REPOSITORY* REPOSITORY;

/// <summary>
///		Creeaza un repository pentru oferte
/// </summary>
/// <param name="Repo">REPOSITORY - repo de creat</param>
void CreateRepo(REPOSITORY* Repo);

/// <summary>
///		Distruge un repository pentru oferte deja creat
/// </summary>
/// <param name="Repo"><REPOSITORY - repo de distrus/param>
void DestroyRepo(REPOSITORY* Repo);

/// <summary>
///		Determina numarul de oferte din repo
/// </summary>
/// <param name="Repo">REPOSITORY - repo caruia ii se determina numarul de oferte</param>
/// <returns>int - numarul de elemente dinr repo</returns>
int GetCountRepo(REPOSITORY Repo);

/// <summary>
///		Adauga o oferta in repo
/// </summary>
/// <param name="Repo">REPOSITORY - repo in care se adauga oferta</param>
/// <param name="Offer">OFFER - oferta de adaugat</param>
/// <returns> 
///  0 - Daca s-a adaugat cu succes oferta
/// -1 - Daca nu se mai pot adauga elemente 
/// -2 - Daca exista deja aceasta oferta in repo
/// </returns>
int AddOfferRepo(REPOSITORY Repo, const OFFER Offer);

/// <summary>
///		Actualizeaza o oferta din repo
/// </summary>
/// <param name="Repo">REPOSITORY - repo in care se actualizeaza oferta</param>
/// <param name="OldOffer">OFFER - oferta care se actualizeaza</param>
/// <param name="NewOffer">OFFER - datele noii oferte</param>
/// <returns>
///  0 - Daca s-a actualizat cu succes
/// -1 - Daca nu exista nicio oferta in repo
/// -3 - Daca nu exista OldOffer in repo
/// </returns>
int UpdateOfferRepo(REPOSITORY Repo, const OFFER OldOffer, const OFFER NewOffer);

/// <summary>
///		Sterge o oferta din repo
/// </summary>
/// <param name="Repo">REPOSITORY - repo in care se sterge oferta</param>
/// <param name="Offer">OFFER - oferta de sters</param>
/// <returns>
///  0 - Daca s-a sters cu succes
/// -1 - Daca nu exista nicio oferta in repo
/// -3 - Daca nu exista oferta in repo
/// </returns>
int DeleteOfferRepo(REPOSITORY Repo, const OFFER Offer);

/// <summary>
///		Verifica daca exista o oferta in repo
/// </summary>
/// <param name="Repo">REPOSITORY - repo in care se cauta</param>
/// <param name="Offer">OFFER - oferta de cautat</param>
/// <returns>
/// 0 - Daca nu exista oferta in repo
/// 1 - Altfel
/// </returns>
BOOL CheckOfferExistsRepo(REPOSITORY Repo, const OFFER Offer);

/// <summary>
///		Getter pentru oferta de pe o anumita index
/// </summary>
/// <param name="Repo">REPOSITORY - repo din care se returneaa o oferta</param>
/// <param name="Index">int - pozitia de pe care sa se returneze oferta</param>
/// <param name="Offer">OFFER - oferta de pe pozitia Index</param>
/// <returns>
///  0 - Daca indexul dat este valid
/// -1 - Altfel
/// </returns>
int GetOfferAfterIndexRepo(REPOSITORY Repo, const int Index, OFFER* Offer);

/// <summary>
///		Determina toate ofertele din repo in Offer si numarul de oferte in Count
/// </summary>
/// <param name="Repo">REPOSITORY - repo din care se returneaza ofertele</param>
/// <param name="Offer">OFFER* - vector cu toate ofertele din repo</param>
/// <param name="Count">int - numarul de oferte din repo</param>
void GetAllOffersRepo(REPOSITORY Repo, OFFER** Offer, int* Count);

#endif // !_REPOSITORY_H_
