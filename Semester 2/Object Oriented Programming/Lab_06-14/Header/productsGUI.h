#pragma once
#include "service.h"
#include "models.h"
#include <QtWidgets/QWidget>
#include <qboxlayout.h>
#include <qlistwidget.h>
#include <qtablewidget.h>
#include <qpushbutton.h>
#include <qlineedit.h>
#include <qlabel.h>
#include <qevent.h>
#include <qpainter.h>
#include <qlistview.h>
#include <vector>
#include <algorithm>
#include <random>

class PaintGUI : public QWidget, public Observer
{
	Q_OBJECT
private:
	ServiceBucket& ServBck;
public:
	PaintGUI(ServiceBucket& _ServBck) : ServBck { _ServBck }
	{
		ServBck.addObserver(this);
		QVBoxLayout* ly = new QVBoxLayout;
		setLayout(ly);
		setFixedWidth(850);
		setFixedHeight(450);
	}
	~PaintGUI()
	{
		ServBck.removeObserver(this);
	}
	void update() override
	{
		repaint();
	}
protected:
	void paintEvent(QPaintEvent* ev) override
	{
		QPainter p{ this };
		std::mt19937 mt{ std::random_device{}() };
		std::uniform_int_distribution<> dist1(0, 800), dist2(0, 400), colors(0, 255);
		int left, top, r, g, b;
		p.eraseRect(0, 0, width(), height());
		for (int i = 0; i < ServBck.getBucket().size(); ++i)
		{
			left = dist1(mt);
			top = dist2(mt);
			r = colors(mt);
			g = colors(mt);
			b = colors(mt);
			p.drawRect(QRect(left, top, 50, 50));
			p.fillRect(left, top, 50, 50, QColor(r, g, b));
		}
	}
};

class BucketGUI : public QWidget, public Observer
{
	Q_OBJECT
	friend class ProductsGUI;
private:
	Service& Serv;
	ServiceBucket& ServBucket;

	ListModel* Model;

	QHBoxLayout* Layout;
	QListView* BucketList;
	//QListWidget* BucketList;
	//QTableWidget* BucketList;
	QListView*& ProductsList;
	//QListWidget*& ProductsList;
	//QTableWidget*& ProductsList;

	QPushButton* BtnAddToBucket;
	QPushButton* BtnClear;
	QPushButton* BtnGenerate;
	QPushButton* BtnExport;
	QPushButton* BtnClearInWindow;
	QPushButton* BtnGenerateInWindow;

	QLineEdit* TextGenerate;
	QLineEdit* TextGenerateInWindow;

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
	void GenerateInWindow();
	void Export();

public:
	BucketGUI(Service& Serv, ServiceBucket& ServBucket, QListView*& ProductsList) : Serv{ Serv }, ServBucket { ServBucket }, ProductsList{ ProductsList }
	//BucketGUI(Service& Serv, ServiceBucket& ServBucket, QListWidget*& ProductsList) : Serv{ Serv }, ServBucket { ServBucket }, ProductsList{ ProductsList }
	//BucketGUI(Service& Serv, ServiceBucket& ServBucket, QTableWidget*& ProductsList) : Serv{ Serv }, ServBucket { ServBucket }, ProductsList{ ProductsList }
	{
		Init();
		ConnectSignals();
		ReloadList(ServBucket.getBucket());
	}

	~BucketGUI() { ServBucket.removeObserver(this); }

	void SetVisible(bool V);
	void reload();
	void connect();

	void update() override;
};

class ProductsGUI : public QWidget
{
	Q_OBJECT
		friend class BucketGUI;
private:
	std::map < std::string, int > evidence;
	Service& Serv;
	ServiceBucket& ServBck;

	ListModel* model;

	//BucketGUI SecondWindow;
	std::vector<BucketGUI*> SecondWindows;
	std::vector<PaintGUI*> PaintWindows;

	QHBoxLayout* MainLayout;
	QVBoxLayout* LeftInfoLayout;

	QListWidget* ListWid;
	QListView* List;

	//QTableWidget* List;

	QWidget* LeftInfoWidget;

	QPushButton* BtnAdd;
	QPushButton* BtnUpdate;
	QPushButton* BtnDelete;
	QPushButton* BtnUndo;
	QPushButton* BtnSortByName;
	QPushButton* BtnSortByPrice;
	QPushButton* BtnSortByNameAndProducer;
	QPushButton* BtnOpenBucket;
	QPushButton* BtnPaintBucket;
	QPushButton* BtnAddToBucket;
	QPushButton* BtnClear;
	QPushButton* BtnGenerate;
	QPushButton* BtnExport;

	QLineEdit* TextId;
	QLineEdit* TextName;
	QLineEdit* TextType;
	QLineEdit* TextProducer;
	QLineEdit* TextPrice;
	QLineEdit* TextFilterName;
	QLineEdit* TextFilterProducer;
	QLineEdit* TextFilterPrice;
	QLineEdit* TextGenerate;

	void Init();
	void ConnectSignals();
	void ReloadList(const std::vector < Product >& Products);
	void AddButtons(const std::vector < Product >& Products);
	void AddWindow();
	void AddPaintWindow();

	void AddProduct();
	void UpdateProduct();
	void DeleteProduct();
	void Undo();

public:
	ProductsGUI(Service& _Serv, ServiceBucket& _ServBck) : Serv{ _Serv }, ServBck{ _ServBck }//, SecondWindow{ _Serv, _ServBck, List }
	{
		Init();
		ConnectSignals();
		ReloadList(Serv.GetAll());
		AddWindow();
	}
};
