#pragma once
#include "service.h"
#include <QtWidgets/QWidget>
#include <qboxlayout.h>
#include <qlistwidget.h>
#include <qpushbutton.h>
#include <qlineedit.h>
#include <qlabel.h>
#include <vector>

class BucketGUI : public QWidget
{
	Q_OBJECT
	friend class ProductsGUI;
private:
	Service& Serv;
	ServiceBucket& ServBucket;

	QHBoxLayout* Layout;
	QListWidget* BucketList;
	QListWidget*& ProductsList;

	QPushButton* BtnAddToBucket;
	QPushButton* BtnClear;
	QPushButton* BtnGenerate;
	QPushButton* BtnExport;

	QLineEdit* TextGenerate;

	QLabel* BucketValue;
	QLabel* IdToShow;
	QLabel* NameToShow;
	QLabel* TypeToShow;
	QLabel* ProducerToShow;
	QLabel* PriceToShow;

	void Init();
	void ConnectSignals();
	void ReloadList(const std::vector < Product >& BucketProducts);

	void AddBucket();
	void Clear();
	void Generate();
	void Export();

public:
	BucketGUI(Service& Serv, ServiceBucket& ServBucket, QListWidget*& ProductsList) : Serv{ Serv }, ServBucket { ServBucket }, ProductsList{ ProductsList }
	{
		Init();
		ConnectSignals();
		ReloadList(ServBucket.getBucket());
	}

	void SetVisible(bool V);
};

class ProductsGUI : public QWidget
{
	Q_OBJECT
	friend class BucketGUI;
private:
	Service& Serv;
	BucketGUI SecondWindow;

	QHBoxLayout* MainLayout;

	QListWidget* List;

	QPushButton* BtnAdd;
	QPushButton* BtnUpdate;
	QPushButton* BtnDelete;
	QPushButton* BtnUndo;
	QPushButton* BtnSortByName;
	QPushButton* BtnSortByPrice;
	QPushButton* BtnSortByNameAndProducer;
	QPushButton* BtnBucket;

	QLineEdit* TextId;
	QLineEdit* TextName;
	QLineEdit* TextType;
	QLineEdit* TextProducer;
	QLineEdit* TextPrice;
	QLineEdit* TextFilterName;
	QLineEdit* TextFilterProducer;
	QLineEdit* TextFilterPrice;

	void Init();
	void ConnectSignals();
	void ReloadList(const std::vector < Product >& Products);

	void AddProduct();
	void UpdateProduct();
	void DeleteProduct();
	void Undo();

public:
	ProductsGUI(Service& Serv, ServiceBucket& ServBck) : Serv{ Serv }, SecondWindow{ Serv, ServBck, List }
	{
		Init();
		ConnectSignals();
		ReloadList(Serv.GetAll());
	}
};
