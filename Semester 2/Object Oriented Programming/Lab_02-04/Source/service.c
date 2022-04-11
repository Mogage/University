#include "common.h"
#include "service.h"
#include "validator.h"

struct _SERVICE 
{
	REPOSITORY Repo;
};

void CreateService(SERVICE* Service, REPOSITORY Repo)
{
	SERVICE temp = NULL;

	temp = (SERVICE)malloc(sizeof(struct _SERVICE));
	if (NULL != temp)
	{
		temp->Repo = Repo;

		*Service = temp;
	}
}

void DestroyService(SERVICE* Service)
{
	SERVICE temp = *Service;

	free(temp);

	*Service = NULL;
}

char* AddOfferService(SERVICE Service, OfferType Type, const char* Destination, struct tm Date, int Price)
{
	OFFER offer = NULL;
	char* errors = NULL;
	int status = -1;

	CreateOffer(&offer, Type, Destination, Date, Price);

	if (FALSE == ValidateOffer(offer, &errors))
	{
		DestroyOffer(&offer);
		return errors;
	}

	status = AddOfferRepo(Service->Repo, offer);

	DestroyOffer(&offer);

	errors = (char*)calloc(101, sizeof(char));
	if (NULL == errors)
	{
		return NULL;
	}

	switch (status)
	{
	case STATUS_EXISTENT:
		strcpy_s(errors, 50, "Aceasta oferta exista deja.\n");
		break;
	default:
		free(errors);
		errors = NULL;
		break;
	}
	
	return errors;
}

char* UpdateOfferService(SERVICE Service, OfferType OldType, const char* OldDestination, struct tm OldDate, OfferType NewType, const char* NewDestination, struct tm NewDate, int NewPrice)
{
	OFFER oldOffer = NULL, newOffer = NULL;
	char* errors = NULL;
	int status = -1;

	if (0 == GetCountRepo(Service->Repo))
	{
		errors = (char*)calloc(101, sizeof(char));
		if (NULL == errors)
		{
			return NULL;
		}

		strcpy_s(errors, 50, "Nu exista nicio oferta.\n");
		return errors;
	}

	CreateOffer(&oldOffer, OldType, OldDestination, OldDate, 1);

	if (FALSE == ValidateOffer(oldOffer, &errors))
	{
		DestroyOffer(&oldOffer);
		return errors;
	}

	CreateOffer(&newOffer, NewType, NewDestination, NewDate, NewPrice);

	if (FALSE == ValidateOffer(newOffer, &errors))
	{
		DestroyOffer(&oldOffer);
		DestroyOffer(&newOffer);
		return errors;
	}	

	status = UpdateOfferRepo(Service->Repo, oldOffer, newOffer);
	
	DestroyOffer(&oldOffer);
	DestroyOffer(&newOffer);

	errors = (char*)calloc(101, sizeof(char));
	if (NULL == errors)
	{
		return NULL;
	}

	switch (status)
	{
	case STATUS_INEXISTENT:
		strcpy_s(errors, 50, "Aceasta oferta nu exista.\n");
		break;
	default:
		free(errors);
		errors = NULL;
		break;
	}


	return errors;
}

char* DeleteOfferService(SERVICE Service, OfferType Type, const char* Destination, struct tm Date)
{
	OFFER offer = NULL;
	char* errors = NULL;
	int status = -1;

	if (0 == GetCountRepo(Service->Repo))
	{
		errors = (char*)calloc(101, sizeof(char));
		if (NULL == errors)
		{
			return NULL;
		}

		strcpy_s(errors, 101, "Nu exista nicio oferta.\n");
		return errors;
	}

	CreateOffer(&offer, Type, Destination, Date, 1);

	if (FALSE == ValidateOffer(offer, &errors))
	{
		DestroyOffer(&offer);
		return errors;
	}

	status = DeleteOfferRepo(Service->Repo, offer);

	DestroyOffer(&offer);

	errors = (char*)calloc(101, sizeof(char));
	if (NULL == errors)
	{
		return NULL;
	}

	switch (status)
	{
	case STATUS_INEXISTENT:
		strcpy_s(errors, 101, "Aceasta oferta nu exista.\n");
		break;
	default:
		free(errors);
		errors = NULL;
		break;
	}

	return errors;
}

BOOL CheckOfferExistsService(SERVICE Service, OfferType Type, const char* Destination, struct tm Date, char** Errors)
{
	OFFER offer = NULL;
	BOOL exists = FALSE;
	char* errors = NULL;

	CreateOffer(&offer, Type, Destination, Date, 1);
	
	if (FALSE == ValidateOffer(offer, &errors))
	{
		DestroyOffer(&offer);
		*Errors = errors;
		return FALSE;
	}

	exists = CheckOfferExistsRepo(Service->Repo, offer);

	DestroyOffer(&offer);

	return exists;
}

int GetCountRepoService(SERVICE Service)
{
	return GetCountRepo(Service->Repo);
}

void GetAllOffersService(SERVICE Service, OFFER** AllOffers, int* Count)
{
	OFFER* offers = NULL;
	int count = 0;

	GetAllOffersRepo(Service->Repo, &offers, &count);

	*AllOffers = offers;
	*Count = count;
}

static void Swap(OFFER* Offer1, OFFER* Offer2)
{
	OFFER aux = *Offer1;
	*Offer1 = *Offer2;
	*Offer2 = aux;
}
/*
static BOOL CompareOrder(int SortType, int ToCheck)
{
	if (1 == SortType)
	{
		return (ToCheck > 0);
	}

	return (ToCheck < 0);
}*/

static BOOL ComparePriceAscending(OFFER Offer1, OFFER Offer2)
{
	return (GetPrice(Offer1) > GetPrice(Offer2));
}

static BOOL ComparePriceDescending(OFFER Offer1, OFFER Offer2)
{
	return (GetPrice(Offer1) < GetPrice(Offer2));
}

static BOOL CompareDestinationAscending(OFFER Offer1, OFFER Offer2)
{
	BOOL toReturn = FALSE;
	char* destination1 = GetDestination(Offer1);
	char* destination2 = GetDestination(Offer2);
	toReturn = (1 == strcmp(destination1, destination2)) ? TRUE : FALSE;
	free(destination1);
	free(destination2);
	return toReturn;
}

static BOOL CompareDestinationDescending(OFFER Offer1, OFFER Offer2)
{
	BOOL toReturn = FALSE;
	char* destination1 = GetDestination(Offer1);
	char* destination2 = GetDestination(Offer2);
	toReturn = (-1 == strcmp(destination1, destination2)) ? TRUE : FALSE;
	free(destination1);
	free(destination2);
	return toReturn;
}

static void SortOffers(OFFER* Offers, int Count, BOOL Compare(OFFER, OFFER))
{
	OFFER* toSort = Offers;

	for (int index1 = 0; index1 < Count - 1; index1 = index1 + 1)
	{
		for (int index2 = index1 + 1; index2 < Count; index2 = index2 + 1)
		{
			if (TRUE == Compare(toSort[index1], toSort[index2]))
			{
				Swap(&toSort[index1], &toSort[index2]);
			}
		}
	}
}
/*
static void SortAfterPrice(OFFER* Offers, int Count, int SortType)
{
	OFFER* toSort = Offers;

	for (int index1 = 0; index1 < Count - 1; index1 = index1 + 1)
	{
		for (int index2 = index1 + 1; index2 < Count; index2 = index2 + 1)
		{
			if (TRUE == CompareOrder(SortType, GetPrice(toSort[index1]) - GetPrice(toSort[index2])))
			{
				Swap(&toSort[index1], &toSort[index2]);
			}
		}
	}

}

static void SortAfterDestination(OFFER* Offers, int Count, int SortType)
{
	OFFER* toSort = Offers;
	char* destination1 = NULL, * destination2 = NULL;

	for (int index1 = 0; index1 < Count - 1; index1 = index1 + 1)
	{
		for (int index2 = index1 + 1; index2 < Count; index2 = index2 + 1)
		{
			destination1 = GetDestination(toSort[index1]);
			destination2 = GetDestination(toSort[index2]);
			if (TRUE == CompareOrder(SortType, strcmp(destination1, destination2)))
			{
				Swap(&toSort[index1], &toSort[index2]);
			}
			free(destination1);
			free(destination2);
		}
	}
}*/

void GetOffersOrderedService(SERVICE Service, int FilterOrder, int TypeOrder, OFFER** Offers, int* Count)
{
	OFFER* offers = NULL;
	int count = 0;

	GetAllOffersRepo(Service->Repo, &offers, &count);

	if (1 == FilterOrder)
	{
		if (1 == TypeOrder)
		{
			SortOffers(offers, count, ComparePriceAscending);
		}
		else
		{
			SortOffers(offers, count, ComparePriceDescending);
		}
		// SortAfterPrice(offers, count, TypeOrder);
	}
	else
	{
		if (1 == TypeOrder)
		{
			SortOffers(offers, count, CompareDestinationAscending);
		}
		else
		{
			SortOffers(offers, count, CompareDestinationDescending);
		}
		// SortAfterDestination(offers, count, TypeOrder);
	}

	*Offers = offers;
	*Count = count;
}

static BOOL CompareFilter(int FilterType, OFFER Offer, const void* Filter)
{
	if (1 == FilterType)
	{
		char* filter = *(char**)Filter;
		char* destination = GetDestination(Offer);
		BOOL returnValue = (0 == strcmp(destination, filter) ? TRUE : FALSE);
		free(destination);
		return returnValue;
	}
	
	int* filter = *(int**)Filter;

	if (2 == FilterType)
	{
		return (*filter == GetType(Offer));
	}

	return (*filter == GetPrice(Offer));
}

void GetOffersFilteredService(SERVICE Service, int FilterType, const void* Filter, OFFER** Offers, int* Count)
{
	OFFER* allOffers = NULL;
	OFFER* filteredOffers = NULL;
	int count = 0, filterCount = 0;
	char* string = NULL;

	GetAllOffersRepo(Service->Repo, &allOffers, &count);

	filteredOffers = (OFFER*)calloc(count, sizeof(OFFER));
	if (NULL != filteredOffers)
	{
		for (int index = 0; index < count; index = index + 1)
		{
			if (TRUE == CompareFilter(FilterType, allOffers[index], Filter))
			{
				if (filterCount < count)
				{
					string = GetDestination(allOffers[index]);
					CreateOffer(&filteredOffers[filterCount], GetType(allOffers[index]), string, GetDate(allOffers[index]), GetPrice(allOffers[index]));
					filterCount = filterCount + 1;
					free(string);
				}
			}
			DestroyOffer(&allOffers[index]);
		}

		free(allOffers);
		*Offers = filteredOffers;
		*Count = filterCount;
	}
}