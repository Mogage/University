#pragma once

#include "domain.h"
#include "repository.h"
#include "service.h"

class Test
{
private:
	void testDomain();
	void testRepo();
	void testServ();
public:
	void run();
};

