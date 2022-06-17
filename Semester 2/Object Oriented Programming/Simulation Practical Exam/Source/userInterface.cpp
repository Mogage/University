#include "userInterface.h"
#include "domain.h"

#include <qlabel.h>
#include <qformlayout.h>
#include <qmessagebox.h>

void UserInterface::Init()
{
	MainLayout = new QHBoxLayout;
	setLayout(MainLayout);

	LeftInfoWidget = new QWidget;
	LeftInfoLayout = new QVBoxLayout;
	LeftInfoWidget->setLayout(LeftInfoLayout);

	List = new QListWidget;

	QWidget* rightSideWidget = new QWidget;
	QFormLayout* rightSideLayout = new QFormLayout;
	QLabel* CodLabel = new QLabel("Cod: ");
	QLabel* BrandLabel = new QLabel("Brand: ");
	QLabel* ModelLabel = new QLabel("Model: ");
	QLabel* PretLabel = new QLabel("Pret: ");
	TextCod = new QLineEdit;
	TextBrand = new QLineEdit;
	TextModel = new QLineEdit;
	TextPret = new QLineEdit;
	ButtonSortBrand = new QPushButton("Sortare dupa brand");
	ButtonSortModel = new QPushButton("Sortare dupa model");
	ButtonSortPrice = new QPushButton("Sortare dupa pret");
	ButtonNotSorted = new QPushButton("Nesortat");

	rightSideWidget->setLayout(rightSideLayout);

	rightSideLayout->addRow(CodLabel, TextCod);
	rightSideLayout->addRow(BrandLabel, TextBrand);
	rightSideLayout->addRow(ModelLabel, TextModel);
	rightSideLayout->addRow(PretLabel, TextPret);
	rightSideLayout->addRow(ButtonSortBrand, ButtonSortModel);
	rightSideLayout->addRow(ButtonSortPrice, ButtonNotSorted);

	MainLayout->addWidget(LeftInfoWidget);
	MainLayout->addWidget(List);
	MainLayout->addWidget(rightSideWidget);
}

void UserInterface::ConnectSignals()
{
	QObject::connect(
		ButtonSortBrand,
		&QPushButton::clicked,
		[&]()
		{
			ReloadList(Serv.getSorted([](const Telefon& Tel1, const Telefon& Tel2) { return Tel1.brand() < Tel2.brand(); }));
		}
	);

	QObject::connect(
		ButtonSortModel,
		&QPushButton::clicked,
		[&]()
		{
			ReloadList(Serv.getSorted([](const Telefon& Tel1, const Telefon& Tel2) { return Tel1.model() < Tel2.model(); }));
		}
	);

	QObject::connect(
		ButtonSortPrice,
		&QPushButton::clicked,
		[&]()
		{
			ReloadList(Serv.getSorted([](const Telefon& Tel1, const Telefon& Tel2) { return Tel1.pret() < Tel2.pret(); }));
		}
	);

	QObject::connect(
		ButtonNotSorted,
		&QPushButton::clicked,
		[&]()
		{
			ReloadList(Serv.getAll());
		}
	);

	QObject::connect(
		List,
		&QListWidget::itemSelectionChanged,
		[&]()
		{
			auto selected = List->selectedItems();
			if (selected.isEmpty())
			{
				TextCod->setText("");
				TextBrand->setText("");
				TextModel->setText("");
				TextPret->setText("");
			}
			else
			{
				auto selectedItem = selected.at(0);
				string toFind = selectedItem->data(Qt::UserRole).toString().toStdString();
				Telefon toDisplay = Serv.findTelefon(toFind);
				TextCod->setText(QString::fromStdString(toDisplay.cod()));
				TextBrand->setText(QString::fromStdString(toDisplay.brand()));
				TextModel->setText(QString::fromStdString(toDisplay.model()));
				TextPret->setText(QString::number(toDisplay.pret()));
			}
		}
	);
}

void ClearLayout(QLayout* Layout)
{
	if (NULL == Layout)
	{
		return;
	}
	QLayoutItem* item;
	while ((item = Layout->takeAt(0)))
	{
		if (item->layout())
		{
			ClearLayout(item->layout());
			delete item->layout();
		}
		if (item->widget())
		{
			delete item->widget();
		}
		delete item;
	}
}

void UserInterface::AddButtons(const vector < Telefon >& Telefoane)
{
	Evidence.clear();
	for (const auto& telefon : Telefoane)
	{
		Evidence[telefon.brand()] = Evidence[telefon.brand()] + 1;
	}

	ClearLayout(LeftInfoLayout);
	for (const auto& type : Evidence)
	{
		QPushButton* toAdd = new QPushButton(QString::fromStdString(type.first));
		LeftInfoLayout->addWidget(toAdd);
		QObject::connect(
			toAdd,
			&QPushButton::clicked,
			[&]() {
				Serv.addTen(type.first);
				QMessageBox::information(NULL, "Info", QString("Numar de telefoane facute de acest brand:") + QString::number(type.second));
			}
		);
	}
}

void UserInterface::ReloadList(const vector<Telefon>& Telefoane)
{
	List->clear();

	for (const auto& telefon : Telefoane)
	{
		string brand = telefon.brand();
		QListWidgetItem* item = new QListWidgetItem(QString::fromStdString(telefon.cod() + " - " + brand + " " + telefon.model()));
		item->setData(Qt::UserRole, std::stoll(telefon.cod()));
		if ("Samsung" == brand)
			item->setBackground(Qt::red);
		else if ("Huawei" == brand)
			item->setBackground(Qt::yellow);
		else if ("Apple" == brand)
			item->setBackground(Qt::blue);
		else
			item->setBackground(Qt::black);

		List->addItem(item);
	}
	AddButtons(Telefoane);
}
