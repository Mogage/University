#include "exceptions.h"
#include "userInterface.h"
#include <qformlayout.h>
#include <qmessagebox.h>
#include <qcolor.h>

void UserInterface::init()
{
	MainLayout = new QHBoxLayout;
	setLayout(MainLayout);

	QWidget* inputWidget = new QWidget;
	QFormLayout* inputLayout = new QFormLayout;
	inputWidget->setLayout(inputLayout);
	TextId = new QLineEdit;
	TextName = new QLineEdit;
	TextType = new QLineEdit;
	TextPrice = new QLineEdit;
	QLabel* idLabel = new QLabel("Id: ");
	QLabel* nameLabel = new QLabel("Name: ");
	QLabel* typeLabel = new QLabel("Type: ");
	QLabel* priceLabel = new QLabel("Price: ");
	inputLayout->addRow(idLabel, TextId);
	inputLayout->addRow(nameLabel, TextName);
	inputLayout->addRow(typeLabel, TextType);
	inputLayout->addRow(priceLabel, TextPrice);

	BtnAdd = new QPushButton("Add product");
	inputLayout->addRow(BtnAdd);

	SliderValue = new QLabel("0");
	Slider = new QSlider(Qt::Horizontal);
	Slider->setTickInterval(1);
	Slider->setMaximum(100);
	inputLayout->addRow(SliderValue, Slider);

	Model = new TableModel{ Serv.all(), *Slider };
	Table = new QTableView;
	Table->setModel(Model);

	MainLayout->addWidget(Table);
	MainLayout->addWidget(inputWidget);
}

void UserInterface::connectSignals()
{
	QObject::connect(Slider, &QSlider::valueChanged, [&]() 
		{ 
			SliderValue->setText(QString::number(Slider->value()));
			Model->update();
		});
	QObject::connect(BtnAdd, &QPushButton::clicked, this, &UserInterface::add);
}

void UserInterface::reloadList(vector<Produs> Products)
{
	Model->setProducts(Products);
	Evidence.clear();
	for (auto& product : Products)
	{
		Evidence[product.type()]++;
	}
	notify();
}

void UserInterface::createWindows()
{
	NumberType* window;
	for (auto& it : Evidence)
	{
		window = new NumberType{ *this, it.first, it.second };
		window->show();
	}
}

void UserInterface::add()
{
	try
	{
		Serv.addProduct(
			TextId->text().toInt(),
			TextName->text().toStdString(),
			TextType->text().toStdString(),
			TextPrice->text().toDouble()
		);
		reloadList(Serv.all());
	}
	catch (const ValidationError& error)
	{
		QMessageBox::warning(this, "Warning", QString::fromStdString(error.getMessage()));
	}
	catch (const RepositoryError& error)
	{
		QMessageBox::warning(this, "Warning", QString::fromStdString(error.getMessage()));
	}
}

UserInterface::UserInterface(Service& _Serv) : Serv{_Serv}
{
	init();
	connectSignals();
	reloadList(Serv.all());
	createWindows();
}

NumberType::NumberType(UserInterface& _Ui, string Name, int _Number) : Ui{ _Ui }
{
	MainLayout = new QHBoxLayout;
	setLayout(MainLayout);
	setWindowTitle(QString::fromStdString(Name));
	setMinimumHeight(100);
	setMinimumWidth(200);

	Number = new QLabel;
	Number->setText("Numar produse de acest tip: " + QString::number(_Number));

	MainLayout->addWidget(Number);
	Ui.addObserver(this);
}

void NumberType::update()
{
	Number->setText("Numar produse de acest tip: " + QString::number(Ui.Evidence[windowTitle().toStdString()]));
}
