#include "common.h"
#include "validator.h"

static BOOL ValidateOfferType(OfferType Type)
{
	return (Type >= 1 && Type <= 3);
}

static BOOL ValidateOfferDate(struct tm Date)
{
	int maxDays = 0;

	if (Date.tm_year < 0)
	{
		return FALSE;
	}
	if (Date.tm_mon < 1 || Date.tm_mon > 12)
	{
		return FALSE;
	}

	switch (Date.tm_mon)
	{
	case 4: case 6: case 9: case 11:
		maxDays = 30;
		break;
	case 2:
		maxDays = 28;
		break;
	default:
		maxDays = 31;
		break;
	}
	
	if (Date.tm_mday < 1 || Date.tm_mday > maxDays)
	{
		return FALSE;
	}

	return TRUE;
}

static BOOL ValidateOfferPrice(int Price)
{
	return (Price >= 0);
}

BOOL ValidateOffer(OFFER Offer, char** Error)
{
	char* errors = NULL;

	errors = (char*)calloc(101, sizeof(char));
	if (NULL == errors)
	{
		return FALSE;
	}

	if (FALSE == ValidateOfferType(GetType(Offer)))
	{
		strcat_s(errors, 101, "Tipul ales este invalid.\n");
	}

	if (FALSE == ValidateOfferDate(GetDate(Offer)))
	{
		strcat_s(errors, 101, "Data introdusa este invalida.\n");
	}

	if (FALSE == ValidateOfferPrice(GetPrice(Offer)))
	{
		strcat_s(errors, 101, "Pretul introdus este invalid.\n");
	}

	if (0 == strlen(errors))
	{
		free(errors);
		return TRUE;
	}

	*Error = errors;
	return FALSE;
}