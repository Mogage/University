#include "common.h"
#include "tests.h"
#include "service.h"
#include <assert.h>

static void TestOffer()
{
	OfferType tip;
	OFFER offer, offer2;
	struct tm timp;
	char* sir = NULL;

	timp.tm_year = 2022;
	timp.tm_mon = 4;
	timp.tm_mday = 25;

	tip = munte;

	CreateOffer(&offer, tip, "Paris", timp, 25);

	assert(GetPrice(offer) == 25);
	// GetDestination(offer, &sir);
	sir = GetDestination(offer);

	assert(strcmp(sir, "Paris") == 0);

	timp = GetDate(offer);
	assert(timp.tm_year == 2022);
	assert(timp.tm_mon == 4);
	assert(timp.tm_mday == 25);
	assert(GetType(offer) == munte);

	free(sir);

	timp.tm_year = 2025;

	SetType(offer, 5);

	sir = PrintOffer(offer);

	assert(NULL == sir);

	SetType(offer, 2);
	SetDestination(offer, "Cluj");
	SetDate(offer, timp);
	SetPrice(offer, 250);

	sir = PrintOffer(offer);

	assert(0 == strcmp(sir, "Tip: mare; Destinatie: Cluj; Data: 25/04/2025; Pret: 250"));

	free(sir);
	sir = NULL;

	CreateOffer(&offer2, 2, "Cluj", timp, 25);

	assert(TRUE == CheckEqual(offer, offer2));

	SetType(offer2, 1);

	assert(FALSE == CheckEqual(offer, offer2));

	sir = PrintOffer(offer2);

	assert(0 == strcmp(sir, "Tip: munte; Destinatie: Cluj; Data: 25/04/2025; Pret: 25"));

	SetType(offer2, city_break);

	free(sir);

	sir = PrintOffer(offer2);

	assert(0 == strcmp(sir, "Tip: city_break; Destinatie: Cluj; Data: 25/04/2025; Pret: 25"));

	SetType(offer2, 2);
	SetDestination(offer2, "Paris");

	assert(FALSE == CheckEqual(offer, offer2));

	SetDestination(offer2, "Cluj");
	timp.tm_mday = 26;
	SetDate(offer2, timp);

	assert(FALSE == CheckEqual(offer, offer2));

	DestroyOffer(&offer);
	DestroyOffer(&offer2);
	free(sir);
}

static void TestRepo()
{
	REPOSITORY repo = NULL;
	OfferType tip;
	OFFER offer, offer2, toUpdateOffer;
	OFFER* vec = NULL;
	struct tm timp;
	char* sir = NULL;
	int count = 0;

	timp.tm_year = 2022;
	timp.tm_mon = 4;
	timp.tm_mday = 25;

	tip = munte;

	CreateOffer(&offer, tip, "Paris", timp, 25);
	CreateOffer(&offer2, mare, "Paris", timp, 25);

	CreateRepo(&repo);

	assert(STATUS_SUCCESS == AddOfferRepo(repo, offer));
	assert(STATUS_SUCCESS == AddOfferRepo(repo, offer2));
	assert(STATUS_EXISTENT == AddOfferRepo(repo, offer));

	DestroyOffer(&offer);

	assert(0 == GetOfferAfterIndexRepo(repo, 0, &offer));
	assert(-1 == GetOfferAfterIndexRepo(repo, 3, &offer));

	GetAllOffersRepo(repo, &vec, &count);

	assert(2 == count);
	assert(TRUE == CheckEqual(vec[0], offer));
	assert(TRUE == CheckEqual(vec[1], offer2));

	assert(TRUE == CheckOfferExistsRepo(repo, offer2));

	SetType(offer2, city_break);

	assert(FALSE == CheckOfferExistsRepo(repo, offer2));
	assert(STATUS_INEXISTENT == DeleteOfferRepo(repo, offer2));
	assert(STATUS_INEXISTENT == UpdateOfferRepo(repo, offer2, offer));

	CreateOffer(&toUpdateOffer, mare, "Paris", timp, 25);

	assert(STATUS_SUCCESS == UpdateOfferRepo(repo, toUpdateOffer, offer2));
	assert(STATUS_SUCCESS == DeleteOfferRepo(repo, offer));

	assert(GetPrice(offer) == 25);
	// GetDestination(offer, &sir);
	sir = GetDestination(offer);
	assert(strcmp(sir, "Paris") == 0);

	timp = GetDate(offer);
	assert(timp.tm_year == 2022);
	assert(timp.tm_mon == 4);
	assert(timp.tm_mday == 25);
	assert(GetType(offer) == munte);
	assert(GetCountRepo(repo) == 1);

	free(sir);
	DestroyOffer(&vec[0]);
	DestroyOffer(&vec[1]);
	free(vec);
	DestroyRepo(&repo);
	DestroyOffer(&offer);
	DestroyOffer(&offer2);
	DestroyOffer(&toUpdateOffer);
}

static void TestValidator()
{
	OFFER offer = NULL;
	struct tm timp;
	char* errors = NULL;

	timp.tm_year = 2020;
	timp.tm_mon = 10;
	timp.tm_mday = 20;

	CreateOffer(&offer, 1, "Japonia", timp, 25);

	assert(TRUE == ValidateOffer(offer, &errors));
	assert(NULL == errors);

	timp.tm_mon = 2;

	SetDate(offer, timp);

	assert(TRUE == ValidateOffer(offer, &errors));
	assert(NULL == errors);

	timp.tm_year = -20;
	timp.tm_mon = 15;
	timp.tm_mday = -25;

	DestroyOffer(&offer);

	CreateOffer(&offer, 5, "Japonia", timp, -235);

	assert(FALSE == ValidateOffer(offer, &errors));
	assert(0 == strcmp(errors, "Tipul ales este invalid.\nData introdusa este invalida.\nPretul introdus este invalid.\n"));



	DestroyOffer(&offer);
	free(errors);
}

static void TestService()
{
	SERVICE service = NULL;
	REPOSITORY repo = NULL;
	OFFER* vec = NULL, *vecAll;
	char* errors = NULL;
	struct tm date;
	int count = 0;

	CreateRepo(&repo);
	CreateService(&service, repo);

	date.tm_year = 2022;
	date.tm_mon = 6;
	date.tm_mday = 5;

	errors = DeleteOfferService(service, city_break, "braila", date);
	assert(0 == strcmp(errors, "Nu exista nicio oferta.\n"));
	free(errors);

	errors = UpdateOfferService(service, city_break, "braila", date, munte, "braila", date, 5);
	assert(0 == strcmp(errors, "Nu exista nicio oferta.\n"));
	free(errors);

	assert(NULL == AddOfferService(service, city_break, "braila", date, 1));

	date.tm_mon = 13;
	errors = DeleteOfferService(service, city_break, "braila", date);
	date.tm_mon = 6;
	assert(0 == strcmp(errors, "Data introdusa este invalida.\n"));
	free(errors);

	errors = DeleteOfferService(service, munte, "braila", date);
	assert(0 == strcmp(errors, "Aceasta oferta nu exista.\n"));
	free(errors);

	errors = UpdateOfferService(service, munte, "braila", date, munte, "braila", date, 5);
	assert(0 == strcmp(errors, "Aceasta oferta nu exista.\n"));
	free(errors);

	errors = UpdateOfferService(service, -5, "braila", date, munte, "braila", date, 5);
	assert(0 == strcmp(errors, "Tipul ales este invalid.\n"));
	free(errors);

	errors = UpdateOfferService(service, munte, "braila", date, -5, "braila", date, 5);
	assert(0 == strcmp(errors, "Tipul ales este invalid.\n"));
	free(errors);

	errors = AddOfferService(service, city_break, "braila", date, 1);
	assert(0 == strcmp(errors, "Aceasta oferta exista deja.\n"));

	date.tm_mon = 1;
	date.tm_mday = 32;

	free(errors);

	errors = AddOfferService(service, city_break, "braila", date, 1);
	assert(0 == strcmp(errors, "Data introdusa este invalida.\n"));
	assert(1 == GetCountRepoService(service));
	free(errors);

	date.tm_mon = 6;
	date.tm_mday = 5;

	assert(TRUE == CheckOfferExistsService(service, city_break, "braila", date, &errors));

	assert(FALSE == CheckOfferExistsService(service, 5, "braila", date, &errors));
	assert(0 == strcmp(errors, "Tipul ales este invalid.\n"));

	GetAllOffersService(service, &vecAll, &count);
	assert(1 == count);
	assert(city_break == GetType(vecAll[0]));
	assert(date.tm_year == GetDate(vecAll[0]).tm_year);
	assert(date.tm_mon == GetDate(vecAll[0]).tm_mon);
	assert(date.tm_mday == GetDate(vecAll[0]).tm_mday);

	DestroyOffer(&vecAll[0]);
	free(vecAll);

	assert(NULL == UpdateOfferService(service, city_break, "braila", date, munte, "braila", date, 5));
	assert(NULL == DeleteOfferService(service, munte, "braila", date));
	assert(0 == GetCountRepoService(service));

	assert(NULL == AddOfferService(service, munte, "brasov", date, 150));
	assert(NULL == AddOfferService(service, mare, "constanta", date, 50));
	assert(NULL == AddOfferService(service, city_break, "cluj", date, 100));
	assert(NULL == AddOfferService(service, munte, "cluj", date, 50));

	GetAllOffersService(service, &vecAll, &count);

	GetOffersOrderedService(service, 1, 1, &vec, &count);
	assert(4 == count);
	assert(TRUE == CheckEqual(vec[0], vecAll[1]));
	assert(TRUE == CheckEqual(vec[1], vecAll[3]));
	assert(TRUE == CheckEqual(vec[2], vecAll[2]));
	assert(TRUE == CheckEqual(vec[3], vecAll[0]));

	for (int i = 0; i < 4; i = i + 1)
	{
		DestroyOffer(&vec[i]);
	}
	free(vec);

	GetOffersOrderedService(service, 1, 2, &vec, &count);
	assert(4 == count);
	assert(TRUE == CheckEqual(vec[0], vecAll[0]));
	assert(TRUE == CheckEqual(vec[1], vecAll[2]));
	assert(TRUE == CheckEqual(vec[2], vecAll[1]));
	assert(TRUE == CheckEqual(vec[3], vecAll[3]));

	for (int i = 0; i < 4; i = i + 1)
	{
		DestroyOffer(&vec[i]);
	}
	free(vec);

	GetOffersOrderedService(service, 2, 1, &vec, &count);
	assert(4 == count);
	assert(TRUE == CheckEqual(vec[0], vecAll[0]));
	assert(TRUE == CheckEqual(vec[1], vecAll[2]));
	assert(TRUE == CheckEqual(vec[2], vecAll[3]));
	assert(TRUE == CheckEqual(vec[3], vecAll[1]));

	for (int i = 0; i < 4; i = i + 1)
	{
		DestroyOffer(&vec[i]);
	}
	free(vec);

	GetOffersOrderedService(service, 2, 2, &vec, &count);
	assert(4 == count);
	assert(TRUE == CheckEqual(vec[0], vecAll[1]));
	assert(TRUE == CheckEqual(vec[1], vecAll[2]));
	assert(TRUE == CheckEqual(vec[2], vecAll[3]));
	assert(TRUE == CheckEqual(vec[3], vecAll[0]));

	for (int i = 0; i < 4; i = i + 1)
	{
		DestroyOffer(&vec[i]);
	}
	free(vec);

	int* filtruInt = (int*)calloc(1, sizeof(int));
	if (NULL == filtruInt)
	{
		return;
	}
	*filtruInt = munte;
	void* filtruGeneric = &filtruInt;

	GetOffersFilteredService(service, 2, filtruGeneric, &vec, &count);
	assert(2 == count);
	assert(TRUE == CheckEqual(vec[0], vecAll[0]));
	assert(TRUE == CheckEqual(vec[1], vecAll[3]));
	
	for (int i = 0; i < 2; i = i + 1)
	{
		DestroyOffer(&vec[i]);
	}
	free(vec);

	*filtruInt = 50;

	GetOffersFilteredService(service, 3, filtruGeneric, &vec, &count);
	assert(2 == count);
	assert(TRUE == CheckEqual(vec[0], vecAll[1]));
	assert(TRUE == CheckEqual(vec[1], vecAll[3]));

	for (int i = 0; i < 2; i = i + 1)
	{
		DestroyOffer(&vec[i]);
	}
	free(vec);

	char* filtruChar = (char*)calloc(50, sizeof(char));
	if (NULL == filtruChar)
	{
		return;
	}
	strcpy_s(filtruChar, 10, "cluj");

	filtruGeneric = &filtruChar;

	GetOffersFilteredService(service, 1, filtruGeneric, &vec, &count);
	assert(2 == count);
	assert(TRUE == CheckEqual(vec[0], vecAll[2]));
	assert(TRUE == CheckEqual(vec[1], vecAll[3]));

	for (int i = 0; i < 4; i = i + 1)
	{
		DestroyOffer(&vec[i]);
		DestroyOffer(&vecAll[i]);
	}
	
	free(filtruInt);
	free(filtruChar);
	free(vec);
	free(vecAll);
	free(errors);
	DestroyRepo(&repo);
	DestroyService(&service);
}


void RunAllTests()
{
	TestOffer();
	TestValidator();
	TestRepo();
	TestService();
}
