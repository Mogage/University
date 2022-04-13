#pragma once

class Test
{
private:
	void TestProduct();

	void TestRepoAdd();
	void TestRepoUpdate();
	void TestRepoFind();
	void TestRepoDelete();
	void TestRepository();

	void TestValidator();

	void TestServAdd();
	void TestServUpdate();
	void TestServFind();
	void TestServDelete();
	void TestFilter();
	void TestSort();
	void TestService();

	void TestBucketRepo();
	void TestBucketService();
	void TestBucket();
public:
	void RunAll();
};

