#include "common.h"
#include "userInterface.h"

struct _USERINT
{
	SERVICE Service;
};

void CreateUserInt(USERINT* UserInt, SERVICE Service)
{
	USERINT temp = NULL;

	temp = (USERINT)malloc(sizeof(struct _USERINT));
	if (NULL == temp)
	{
		return;
	}

	temp->Service = Service;

	*UserInt = temp;
}

void DestroyUserInt(USERINT* UserInt)
{
	USERINT temp = *UserInt;

	free(temp);

	*UserInt = NULL;
}

static void PrintMenu()
{
	printf(" Meniu:\n"
		   "\t 1. Introdu 1 pentru a adauga o oferta.\n"
		   "\t 2. Introdu 2 pentru a actualiza o oferta.\n"
		   "\t 3. Introdu 3 pentru a sterge o oferta.\n"
		   "\t 4. Introdu 4 pentru a afisa toate ofertele.\n"
		   "\t 5. Introdu 5 pentru a vizualiza ofertele ordonat.\n"
		   "\t 6. Introdu 6 pentru a vizualiza ofertele dupa un filtru.\n"
		   "\t 7. Introdu 7 pentru a afisa meniul.\n"
		   "\t 8. Introdu 0 pentru a iesi din aplicatie.\n");
}

static void AddOfferUserInt(USERINT UserInt)
{
	OfferType type;
	char* string = NULL, * errors = NULL;
	struct tm date;
	int price, maxDays, error;
	size_t foundPosition;

	string = (char*)calloc(256, sizeof(char));
	if (NULL == string)
	{
		return;
	}

readTypeAdd:
	printf("Alegeti tipul calatoriei: 1 - munte; 2 - mare; 3 - city break.\n>>>");
	if (scanf_s("%d", &type) == 0)
	{
		fseek(stdin, 0L, SEEK_END);
		goto readTypeAdd;
	}

	error = getchar();
readDestinationAdd:
	printf("Introduceti destinatia(maxim 255 caractere): ");
	fgets(string, 255, stdin);
	foundPosition = strcspn(string, "\n");
	if (0 == foundPosition)
	{
		goto readDestinationAdd;
	}
	string[foundPosition] = 0;
	fseek(stdin, 0L, SEEK_END);

readYearAdd:
	printf("Introduceti anul plecarii: ");
	if (scanf_s("%d", &date.tm_year) == 0)
	{
		fseek(stdin, 0L, SEEK_END);
		goto readYearAdd;
	}

readMonthAdd:
	printf("Introduceti luna plecarii(1-12): ");
	if (scanf_s("%d", &date.tm_mon) == 0)
	{
		fseek(stdin, 0L, SEEK_END);
		goto readMonthAdd;
	}

	switch (date.tm_mon)
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

readDayAdd:
	printf("Introduceti ziua plecarii(1-%d): ", maxDays);
	if (scanf_s("%d", &date.tm_mday) == 0)
	{
		fseek(stdin, 0L, SEEK_END);
		goto readDayAdd;
	}
	
readPriceAdd:
	printf("Introduceti pretul: ");
	if (scanf_s("%d", &price) == 0)
	{
		fseek(stdin, 0L, SEEK_END);
		goto readPriceAdd;
	}

	error = getchar();
	errors = AddOfferService(UserInt->Service, type, string, date, price);
	if (NULL != errors)
	{
		printf("%s", errors);
		free(errors);
	}
	else
	{
		printf("Oferta adaugata cu succes.\n");
	}
	free(string);
}

static void ReadOldOfferUpdateUserInt(OfferType* Type, char** Destination, struct tm* Date)
{
	OfferType type;
	char* string = NULL;
	struct tm date;
	int maxDays, error;
	size_t foundPosition;

	string = (char*)calloc(256, sizeof(char));
	if (NULL == string)
	{
		return;
	}

readTypeOldUpdate:
	printf("Alegeti tipul calatoriei ofertei de actualizat: 1 - munte; 2 - mare; 3 - city break.\n>>>");
	if (scanf_s("%d", &type) == 0)
	{
		fseek(stdin, 0L, SEEK_END);
		goto readTypeOldUpdate;
	}
	error = getchar();
readDestinationOldUpdate:
	printf("Introduceti destinatia ofertei de actualizat(maxim 255 caractere): ");
	fgets(string, 255, stdin);
	foundPosition = strcspn(string, "\n");
	if (0 == foundPosition)
	{
		goto readDestinationOldUpdate;
	}
	string[foundPosition] = 0;
	fseek(stdin, 0L, SEEK_END);

readYearOldUpdate:
	printf("Introduceti anul plecarii ofertei de actualizat: ");
	if (scanf_s("%d", &date.tm_year) == 0)
	{
		fseek(stdin, 0L, SEEK_END);
		goto readYearOldUpdate;
	}

readMonthOldUpdate:
	printf("Introduceti luna plecarii ofertei de actualizat(1-12): ");
	if (scanf_s("%d", &date.tm_mon) == 0)
	{
		fseek(stdin, 0L, SEEK_END);
		goto readMonthOldUpdate;
	}

	switch (date.tm_mon)
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

readDayOldUpdate:
	printf("Introduceti ziua plecarii ofertei de actualizat(1-%d): ", maxDays);
	if (scanf_s("%d", &date.tm_mday) == 0)
	{
		fseek(stdin, 0L, SEEK_END);
		goto readDayOldUpdate;
	}

	error = getchar();
	*Type = type;
	*Destination = string;
	*Date = date;
}

static void ReadNewOfferUpdateUserInt(OfferType* Type, char** Destination, struct tm* Date, int* Price)
{
	OfferType type;
	char* string = NULL;
	struct tm date;
	int price, maxDays, error;
	size_t foundPosition;

	string = (char*)calloc(256, sizeof(char));
	if (NULL == string)
	{
		return;
	}

readTypeNewUpdate:
	printf("Alegeti noul tip al calatoriei: 1 - munte; 2 - mare; 3 - city break.\n>>>");
	if (scanf_s("%d", &type) == 0)
	{
		fseek(stdin, 0L, SEEK_END);
		goto readTypeNewUpdate;
	}

	error = getchar();
readDestinationNewUpdate:
	printf("Introduceti noua destinatia(maxim 255 caractere): ");
	fgets(string, 255, stdin);
	foundPosition = strcspn(string, "\n");
	if (0 == foundPosition)
	{
		goto readDestinationNewUpdate;
	}
	string[foundPosition] = 0;
	fseek(stdin, 0L, SEEK_END);

readYearNewUpdate:
	printf("Introduceti noul anul de plecare: ");
	if (scanf_s("%d", &date.tm_year) == 0)
	{
		fseek(stdin, 0L, SEEK_END);
		goto readYearNewUpdate;
	}

readMonthNewUpdate:
	printf("Introduceti noua luna de plecare(1-12): ");
	if (scanf_s("%d", &date.tm_mon) == 0)
	{
		fseek(stdin, 0L, SEEK_END);
		goto readMonthNewUpdate;
	}

	switch (date.tm_mon)
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

readDayNewUpdate:
	printf("Introduceti noua zi de plecare(1-%d): ", maxDays);
	if (scanf_s("%d", &date.tm_mday) == 0)
	{
		fseek(stdin, 0L, SEEK_END);
		goto readDayNewUpdate;
	}

readPriceNewUpdate:
	printf("Introduceti noul pret: ");
	if (scanf_s("%d", &price) == 0)
	{
		fseek(stdin, 0L, SEEK_END);
		goto readPriceNewUpdate;
	}

	error = getchar();

	*Type = type;
	*Destination = string;
	*Date = date;
	*Price = price;
}

static void UpdateOfferUserInt(USERINT UserInt)
{
	OfferType oldType, newType;
	char* oldDestination = NULL, * newDestination = NULL, * errors = NULL;
	struct tm oldDate = { 0 }, newDate;
	int newPrice;
	BOOL exists = TRUE;

	if (0 == GetCountRepoService(UserInt->Service))
	{
		printf("Nu exista nicio oferta.\n");
		return;
	}

	ReadOldOfferUpdateUserInt(&oldType, &oldDestination, &oldDate);

	exists = CheckOfferExistsService(UserInt->Service, oldType, oldDestination, oldDate, &errors);
	if (FALSE == exists)
	{
		if (NULL != errors)
		{
			printf("%s", errors);
			free(errors);
		}
		else
		{
			printf("Nu exista aceasta oferta.\n");
		}

		return;
	}

	ReadNewOfferUpdateUserInt(&newType, &newDestination, &newDate, &newPrice);

	errors = UpdateOfferService(UserInt->Service, oldType, oldDestination, oldDate, newType, newDestination, newDate, newPrice);

	if (NULL != errors)
	{
		printf("%s", errors);
		free(errors);
	}
	else
	{
		printf("Oferta actualizata cu succes.\n");
	}
	free(oldDestination);
	free(newDestination);
}

static void DeleteOfferUserInt(USERINT UserInt)
{
	OfferType type;
	char* string = NULL, * errors = NULL;
	struct tm date;
	int maxDays, error;
	size_t foundPosition;

	if (0 == GetCountRepoService(UserInt->Service))
	{
		printf("Nu exista nicio oferta.\n");
		return;
	}

	string = (char*)calloc(256, sizeof(char));
	if (NULL == string)
	{
		return;
	}

readTypeDelete:
	printf("Alegeti tipul calatoriei ofertei de sters: 1 - munte; 2 - mare; 3 - city break.\n>>>");
	if (scanf_s("%d", &type) == 0)
	{
		fseek(stdin, 0L, SEEK_END);
		goto readTypeDelete;
	}
	
	error = getchar();
readDestinationDelete:
	printf("Introduceti destinatia ofertei de sters(maxim 255 caractere): ");
	fgets(string, 255, stdin);
	foundPosition = strcspn(string, "\n");
	if (0 == foundPosition)
	{
		goto readDestinationDelete;
	}
	string[foundPosition] = 0;
	fseek(stdin, 0L, SEEK_END);

readYearDelete:
	printf("Introduceti anul plecarii ofertei de sters: ");
	if (scanf_s("%d", &date.tm_year) == 0)
	{
		fseek(stdin, 0L, SEEK_END);
		goto readYearDelete;
	}

readMonthDelete:
	printf("Introduceti luna plecarii ofertei de sters(1-12): ");
	if (scanf_s("%d", &date.tm_mon) == 0)
	{
		fseek(stdin, 0L, SEEK_END);
		goto readMonthDelete;
	}

	switch (date.tm_mon)
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

readDayDelete:
	printf("Introduceti ziua plecarii ofertei de sters(1-%d): ", maxDays);
	if (scanf_s("%d", &date.tm_mday) == 0)
	{
		fseek(stdin, 0L, SEEK_END);
		goto readDayDelete;
	}

	error = getchar();
	errors = DeleteOfferService(UserInt->Service, type, string, date);
	if (NULL != errors)
	{
		printf("%s", errors);
		free(errors);
	}
	else
	{
		printf("Oferta stearsa cu succes.\n");
	}
	free(string);
}

static void PrintAllOffers(USERINT UserInt)
{
	OFFER* allOffers = NULL;
	int count = 0;
	char* toPrint = NULL;
	
	if (0 == GetCountRepoService(UserInt->Service))
	{
		printf("Nu exista nicio oferta.\n");
		return;
	}

	GetAllOffersService(UserInt->Service, &allOffers, &count);

	for (int index = 0; index < count; index = index + 1)
	{
		toPrint = PrintOffer(allOffers[index]);
		printf("%s\n", toPrint);
		free(toPrint);
		DestroyOffer(&allOffers[index]);
	}
	free(allOffers);
}

static void PrintOrderedOffers(USERINT UserInt)
{
	OFFER* orderedOffers = NULL;
	int count = 0, filterOrder = 0, typeOrder = 0;
	char* toPrint = NULL;

	if (0 == GetCountRepoService(UserInt->Service))
	{
		printf("Nu exista nicio oferta.\n");
		return;
	}

readOrderFilter:
	printf("Ordonati dupa pret(1) / destinatie(2) sau 0 pentru a iesi: ");
	if (scanf_s("%d", &filterOrder) == 0)
	{
		fseek(stdin, 0L, SEEK_END);
		goto readOrderFilter;
	}

	if (0 == filterOrder)
	{
		return;
	}
	else if (filterOrder < 1 || filterOrder > 2)
	{
		fseek(stdin, 0L, SEEK_END);
		goto readOrderFilter;
	}

readTypeOrder:
	printf("Ordonati crescator(1) / descrescator(2) sau 0 pentru a iesi : ");
	if (scanf_s("%d", &typeOrder) == 0)
	{
		fseek(stdin, 0L, SEEK_END);
		goto readTypeOrder;
	}

	if (0 == typeOrder)
	{
		return;
	}
	else if (typeOrder < 1 || typeOrder > 2)
	{
		fseek(stdin, 0L, SEEK_END);
		goto readTypeOrder;
	}

	GetOffersOrderedService(UserInt->Service, filterOrder, typeOrder, &orderedOffers, &count);

	for (int index = 0; index < count; index = index + 1)
	{
		toPrint = PrintOffer(orderedOffers[index]);
		printf("%s\n", toPrint);
		free(toPrint);
		DestroyOffer(&orderedOffers[index]);
	}
	free(orderedOffers);
}

static void PrintFilteredOffers(USERINT UserInt)
{
	OFFER* filteredOffers = NULL;
	int count = 0, filterType = 0, error;
	void* generalFilter = NULL;
	char* toPrint = NULL;

	if (0 == GetCountRepoService(UserInt->Service))
	{
		printf("Nu exista nicio oferta.\n");
		return;
	}

readFilterType:
	printf( "Filtrati dupa:\n"
			"\t 1 - destinatie\n"
			"\t 2 - tip\n"
			"\t 3 - pret\n"
			"\t 0 - pentru a iesi\n");
	if (scanf_s("%d", &filterType) == 0)
	{
		fseek(stdin, 0L, SEEK_END);
		goto readFilterType;
	}

	if (0 == filterType)
	{
		return;
	}
	else if (filterType < 1 || filterType > 3)
	{
		fseek(stdin, 0L, SEEK_END);
		goto readFilterType;
	}

	if (1 == filterType)
	{
		char* filter;
		size_t foundPosition;

		filter = (char*)calloc(255, sizeof(char));
		if (NULL == filter)
		{
			return;
		}

		error = getchar();
	readStringFilter:
		printf("Introduceti destinatia(maxim 255 caractere): ");
		fgets(filter, 255, stdin);
		foundPosition = strcspn(filter, "\n");
		if (0 == foundPosition)
		{
			goto readStringFilter;
		}
		filter[foundPosition] = 0;
		fseek(stdin, 0L, SEEK_END);
		generalFilter = &filter;
	}
	else
	{
		int* filter;

		filter = (int*)calloc(1, sizeof(int));
		if (NULL == filter)
		{
			return;
		}

	readIntFilter:
		if (2 == filterType)
		{
			printf("Alegeti tipul:1 - munte; 2 - mare; 3 - city break: ");
		}
		else
		{
			printf("Introduceti pretul: ");
		}
		if (scanf_s("%d", filter) == 0)
		{
			fseek(stdin, 0L, SEEK_END);
			goto readIntFilter;
		}
		generalFilter = &filter;
	}

	GetOffersFilteredService(UserInt->Service, filterType, generalFilter, &filteredOffers, &count);

	for (int index = 0; index < count; index = index + 1)
	{
		toPrint = PrintOffer(filteredOffers[index]);
		printf("%s\n", toPrint);
		free(toPrint);
		DestroyOffer(&filteredOffers[index]);
	}
	if (1 == filterType)
	{
		free(*(char**)generalFilter);
	}
	else
	{
		free(*(int**)generalFilter);
	}
	free(filteredOffers);
}

void RunApp(USERINT UserInt)
{
	char userInput[255];

	PrintMenu();
	while (TRUE)
	{
		fseek(stdin, 0L, SEEK_END);
		printf(">>>");
		fgets(userInput, 255, stdin);
		userInput[strcspn(userInput, "\n")] = 0;
		
		if (0 == strcmp(userInput, "0"))
		{
			printf("Aplicatia s-a incheiat cu succes.");
			return;
		}
		else if (0 == strcmp(userInput, ""))
		{
			continue;
		}

		if (0 == strcmp(userInput, "1"))
		{
			AddOfferUserInt(UserInt);
		}
		else if (0 == strcmp(userInput, "2"))
		{
			UpdateOfferUserInt(UserInt);
		}
		else if (0 == strcmp(userInput, "3"))
		{
			DeleteOfferUserInt(UserInt);
		}
		else if (0 == strcmp(userInput, "4"))
		{
			PrintAllOffers(UserInt);
		}
		else if (0 == strcmp(userInput, "5"))
		{
			PrintOrderedOffers(UserInt);
		}
		else if (0 == strcmp(userInput, "6"))
		{
			PrintFilteredOffers(UserInt);
		}
		else if (0 == strcmp(userInput, "7"))
		{
			PrintMenu();
		}
		else
		{
			printf("Comanda invalida.\n");
		}
		
	}
}