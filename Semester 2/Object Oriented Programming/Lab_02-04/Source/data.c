#include "common.h"
#include "data.h"

struct _OFFER 
{
	OfferType Type;
	char* Destination;
	struct tm Date;
	int Price;
};

void CreateOffer(OFFER* Offer, OfferType Type, const char* Destination, struct tm Date, int Price)
{
	OFFER temp = NULL;

	temp = (OFFER)malloc(sizeof(struct _OFFER));
	if (NULL != temp)
	{
		temp->Type = Type;
		temp->Destination = (char*)calloc(strlen(Destination) + 1, sizeof(char));
		if (NULL != temp->Destination)
		{
			strcpy_s(temp->Destination, strlen(Destination) + 1, Destination);
			temp->Date = Date;
			temp->Price = Price;

			*Offer = temp;
		}
	}
}

void DestroyOffer(OFFER* Offer)
{
	OFFER temp = *Offer;

	if (NULL == *Offer || NULL == Offer)
	{
		return;
	}

	free(temp->Destination);
	temp->Destination = NULL;
	free(temp);

	*Offer = NULL;
}

void SetDate(OFFER Offer, struct tm Date)
{
	Offer->Date = Date;
}

void SetPrice(OFFER Offer, int Price)
{
	Offer->Price = Price;
}

void SetDestination(OFFER Offer, const char* Destination)
{
	if (NULL != Offer->Destination)
	{
		free(Offer->Destination);
	}
	Offer->Destination = (char*)calloc(strlen(Destination) + 1, sizeof(char));
	if (NULL != Offer->Destination)
	{
		strcpy_s(Offer->Destination, strlen(Destination) + 1, Destination);
	}
}

void SetType(OFFER Offer, OfferType Type)
{
	Offer->Type = Type;
}

struct tm GetDate(OFFER Offer)
{
	return Offer->Date;
}

int GetPrice(OFFER Offer)
{
	return Offer->Price;
}

char* GetDestination(OFFER Offer)
{
	char* Destination;

	Destination = (char*)calloc(strlen(Offer->Destination) + 1, sizeof(char));
	if (NULL == Destination)
	{
		return NULL;
	}

	strcpy_s(Destination, strlen(Offer->Destination) + 1, Offer->Destination);

	// *ReturnedValue = Destination;
	return Destination;
}

OfferType GetType(OFFER Offer)
{
	return Offer->Type;
}

BOOL CheckEqual(OFFER Offer1, OFFER Offer2)
{
	if (Offer1->Type != Offer2->Type)
	{
		return FALSE;
	}

	if (0 != strcmp(Offer1->Destination, Offer2->Destination))
	{
		return FALSE;
	}

	if (Offer1->Date.tm_year != Offer2->Date.tm_year || 
		Offer1->Date.tm_mon  != Offer2->Date.tm_mon  || 
		Offer1->Date.tm_mday != Offer2->Date.tm_mday)

	{
		return FALSE;
	}

	return TRUE;
}

char* PrintOffer(OFFER Offer)
{
	char* format = NULL;

	format = (char*)calloc(150, sizeof(char));
	if (NULL == format)
	{
		return NULL;
	}

	strcpy_s(format, 150, "Tip: ");

	switch (Offer->Type)
	{
	case 1:
		strcat_s(format, 150, "munte");
		break;
	case 2:
		strcat_s(format, 150, "mare");
		break;
	case 3:
		strcat_s(format, 150, "city_break");
		break;
	default:
		free(format);
		return NULL;
	}

	// snprintf(format + strlen(format), 100, "; Destinatie: %s; Data: %02d/%02d/%d; Pret: %d", Offer->Destination, Offer->Date.tm_mday, Offer->Date.tm_mon, Offer->Date.tm_year, Offer->Price);
	sprintf_s(format + strlen(format), 150 - strlen(format), "; Destinatie: %s; Data: %02d/%02d/%d; Pret: %d", Offer->Destination, Offer->Date.tm_mday, Offer->Date.tm_mon, Offer->Date.tm_year, Offer->Price);
	return format;
}