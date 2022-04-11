#include "common.h"
#include "repository.h"

struct _REPOSITORY 
{
	OFFER* Array;
	int Size;
	int Count;
};

/*
static void LoadFromFile(OFFER** Array, int Count)
{
	FILE* stream = NULL;
	OFFER* offersFromFile = NULL;
	char* buffer = NULL;

	buffer = (char*)calloc(1000, sizeof(char));
	if (NULL == buffer)
	{
		return;
	}

	stream = fopen("oferte.txt", "r");
	if (NULL == stream)
	{
		free(buffer);
		return;
	}

	fgets(buffer, 255, stream);

	printf("%s", buffer);

	free(buffer);
	fclose(stream);
}
*/

void CreateRepo(REPOSITORY* Repo)
{
	REPOSITORY temp = NULL;

	temp = (REPOSITORY)malloc(sizeof(struct _REPOSITORY));
	if (NULL != temp)
	{
		temp->Array = (OFFER*)calloc(INITIAL_SIZE, sizeof(OFFER));
		if (NULL != temp->Array)
		{
			temp->Count = 0;
			temp->Size = INITIAL_SIZE;

			// LoadFromFile(&temp->Array, &temp->Count);

			*Repo = temp;
		}
	}
}

void DestroyRepo(REPOSITORY* Repo)
{
	REPOSITORY temp = *Repo;

	for (int i = 0; i < temp->Count; i = i + 1)
	{
		DestroyOffer(&temp->Array[i]);
		temp->Array[i] = NULL;
	}
	free(temp->Array);
	temp->Array = NULL;
	free(temp);
	
	*Repo = NULL;
}

int GetCountRepo(REPOSITORY Repo)
{
	return Repo->Count;
}

int AddOfferRepo(REPOSITORY Repo, OFFER Offer)
{
	char* sir = NULL;

	for (int index = 0; index < Repo->Count; index = index + 1)
	{
		if (TRUE == CheckEqual(Offer, Repo->Array[index]))
		{
			return STATUS_EXISTENT;
		}
	}

	sir = GetDestination(Offer);
	CreateOffer(&Repo->Array[Repo->Count], GetType(Offer), sir, GetDate(Offer), GetPrice(Offer));

	Repo->Count = Repo->Count + 1;

	free(sir);

	return STATUS_SUCCESS;
}

int UpdateOfferRepo(REPOSITORY Repo, const OFFER OldOffer, const OFFER NewOffer)
{
	int index;
	char* sir = NULL;

	for (index = 0; index < Repo->Count; index = index + 1)
	{
		if (TRUE == CheckEqual(OldOffer, Repo->Array[index]))
		{
			SetType(Repo->Array[index], GetType(NewOffer));
			sir = GetDestination(NewOffer);
			SetDestination(Repo->Array[index], sir);
			SetDate(Repo->Array[index], GetDate(NewOffer));
			SetPrice(Repo->Array[index], GetPrice(NewOffer));
			break;
		}
	}
	free(sir);

	if (index == Repo->Count)
	{
		return STATUS_INEXISTENT;
	}

	return STATUS_SUCCESS;
}

int DeleteOfferRepo(REPOSITORY Repo, const OFFER Offer)
{
	int index;
	char* sir = NULL;

	for (index = 0; index < Repo->Count; index = index + 1)
	{
		if (TRUE == CheckEqual(Offer, Repo->Array[index]))
		{
			// DestroyOffer(&Repo->Array[index]);
			break;
		}
	}

	if (index == Repo->Count)
	{
		return STATUS_INEXISTENT;
	}

	for (; index < Repo->Count - 1; index = index + 1)
	{
		sir = GetDestination(Repo->Array[index + 1]);
		SetDestination(Repo->Array[index], sir);
		SetType(Repo->Array[index], GetType(Repo->Array[index + 1]));
		SetDate(Repo->Array[index], GetDate(Repo->Array[index + 1]));
		SetPrice(Repo->Array[index], GetPrice(Repo->Array[index + 1]));
	}

	if (Repo->Array[index] != NULL)
	{
		DestroyOffer(&Repo->Array[index]);
	}

	Repo->Count = Repo->Count - 1;

	free(sir);

	return STATUS_SUCCESS;
}

BOOL CheckOfferExistsRepo(REPOSITORY Repo, const OFFER Offer)
{
	for (int index = 0; index < Repo->Count; index = index + 1)
	{
		if (TRUE == CheckEqual(Repo->Array[index], Offer))
		{
			return TRUE;
		}
	}
	
	return FALSE;
}

int GetOfferAfterIndexRepo(REPOSITORY Repo, const int Index, OFFER* Offer)
{
	OFFER temp = NULL;
	char* sir = NULL;

	if (Index < 0 || Index >= Repo->Count)
	{
		return STATUS_ERROR;
	}

	// GetDestination(Repo->Array[Index], &sir);
	sir = GetDestination(Repo->Array[Index]);
	CreateOffer(&temp, GetType(Repo->Array[Index]), sir, GetDate(Repo->Array[Index]), GetPrice(Repo->Array[Index]));

	*Offer = temp;

	free(sir);
	return STATUS_SUCCESS;
}

void GetAllOffersRepo(REPOSITORY Repo, OFFER** Offer, int* Count)
{
	OFFER* temp = NULL;

	temp = (OFFER*)calloc(Repo->Count, sizeof(OFFER));
	if (temp != NULL)
	{
		for (int index = 0; index < Repo->Count; index = index + 1)
		{
			GetOfferAfterIndexRepo(Repo, index, &temp[index]);
		}

		*Offer = temp;
		*Count = Repo->Count;
	}
}