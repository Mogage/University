#ifndef _DATA_H_
#define _DATA_H_

typedef enum { munte = 1, mare, city_break } OfferType;

typedef struct _OFFER* OFFER;

/// <summary>
///		Creeaza o noua oferta cu datele specificate
/// </summary>
/// <param name="Offer">OFFER - Oferta de creat</param>
/// <param name="Type">OfferType - Tipul ofertei</param>
/// <param name="Destination">char* - Destinatia ofertei</param>
/// <param name="Date">struct tm - Data ofertei</param>
/// <param name="Price">int - Pretul ofertei</param>
void CreateOffer(OFFER* Offer, OfferType Type, const char* Destination, struct tm Date, int Price);

/// <summary>
///		Distruge o oferta deja creata
/// </summary>
/// <param name="Offer">OFFER - Oferta de distrus</param>
void DestroyOffer(OFFER* Offer);

/// <summary>
///		Seteaza data unei oferte
/// </summary>
/// <param name="Offer">OFFER - Oferta caruia ii se schimba data</param>
/// <param name="Date">struct tm - Noua data</param>
void SetDate(OFFER Offer, struct tm Date);

/// <summary>
///		Seteaza pretul unei oferte
/// </summary>
/// <param name="Offer">OFFER - Oferta caruia ii se schimba pretul</param>
/// <param name="Price">int - Noul pret</param>
void SetPrice(OFFER Offer, int Price);

/// <summary>
///		Seteaza destinatia unei oferte
/// </summary>
/// <param name="Offer">OFFER - Oferta caruia ii se schimba destinatia</param>
/// <param name="Destination">char* - Noua destinatie</param>
void SetDestination(OFFER Offer, const char* Destination);

/// <summary>
///		Seteaza tipul unei oferte
/// </summary>
/// <param name="Offer">OFFER - Oferta caruia ii se schimba tipul</param>
/// <param name="Type">OfferType - Noul tip</param>
void SetType(OFFER Offer, OfferType Type);

/// <summary>
///		Getter pentru data unei oferte
/// </summary>
/// <param name="Offer">OFFER - Oferta pentru care se returneaza data</param>
/// <returns>struct tm - data ofertei</returns>
struct tm GetDate(OFFER Offer);

/// <summary>
///		Getter pentru pretul unei oferte
/// </summary>
/// <param name="Offer">OFFER - Oferta pentru care se returneaza pretul</param>
/// <returns>int - pretul ofertei</returns>
int GetPrice(OFFER Offer);

/// <summary>
///		Getter pentru destinatia unei oferte
///		Este nevoie de dealocat memoria pentru sir de catre apelant
/// </summary>
/// <param name="Offer">OFFER - Oferta pentru se returneaza destinatia</param>
/// <returns>char* - destinatia ofertei</returns>
char* GetDestination(OFFER Offer);

/// <summary>
///		Getter pentru tipul unei oferte
/// </summary>
/// <param name="Offer">OFFER - Oferta pentru care returneaza tipul</param>
/// <returns>OfferType - tipul ofertei</returns>
OfferType GetType(OFFER Offer);

/// <summary>
///		Verifica daca 2 oferte sunt egale
/// </summary>
/// <param name="Offer1">OFFER</param>
/// <param name="Offer2">OFFER</param>
/// <returns>true, daca ofertele sunt egale, false altfel</returns>
BOOL CheckEqual(OFFER Offer1, OFFER Offer2);

/// <summary>
///		Formateaza o oferta pentru afisare
/// </summary>
/// <param name="Offer">OFFER - Oferta de afisat</param>
/// <returns>char* - sirul care contine formatul de afisare al ofertei</returns>
char* PrintOffer(OFFER Offer);

#endif // !_DATA_H_
