#include "productsGUI.h"
#include "exceptions.h"
#include <qformlayout.h>
#include <qmessagebox.h>
#include <qgridlayout.h>
#include <qfiledialog.h>
#include <qfile.h>

void ProductsGUI::Init()
{
	MainLayout = new QHBoxLayout;
	setLayout(MainLayout);
	// setStyleSheet("background-color: black;");

	QWidget* printingWidget = new QWidget;
	QVBoxLayout* printingLayout = new QVBoxLayout;
	printingWidget->setLayout(printingLayout);
	List = new QListWidget;

	QWidget* gridButtonsWidget = new QWidget;
	QGridLayout* gridButtonsLayout = new QGridLayout;
	gridButtonsWidget->setLayout(gridButtonsLayout);
	BtnSortByName = new QPushButton("Sort by name");
	BtnSortByPrice = new QPushButton("Sort by price");
	BtnSortByNameAndProducer = new QPushButton("Sort by name and producer");
	QLabel* filterByNameLabel = new QLabel("Filter by name: ");
	QLabel* filterByProducerLabel = new QLabel("Filter by Producer: ");
	QLabel* filterByPriceLabel = new QLabel("Filter by Price: ");
	TextFilterName = new QLineEdit;
	TextFilterProducer = new QLineEdit;
	TextFilterPrice = new QLineEdit;
	gridButtonsLayout->addWidget(BtnSortByName, 0, 0);
	gridButtonsLayout->addWidget(BtnSortByPrice, 1, 0);
	gridButtonsLayout->addWidget(BtnSortByNameAndProducer, 2, 0);
	gridButtonsLayout->addWidget(filterByNameLabel, 0, 1);
	gridButtonsLayout->addWidget(filterByProducerLabel, 1, 1);
	gridButtonsLayout->addWidget(filterByPriceLabel, 2, 1);
	gridButtonsLayout->addWidget(TextFilterName, 0, 2);
	gridButtonsLayout->addWidget(TextFilterProducer, 1, 2);
	gridButtonsLayout->addWidget(TextFilterPrice, 2, 2);

	// List->setStyleSheet("QListWidget{ background: grey; }");
	printingLayout->addWidget(List);
	printingLayout->addWidget(gridButtonsWidget);

	QWidget* formDataWidget = new QWidget;
	QFormLayout* formDataLayout = new QFormLayout;
	formDataWidget->setLayout(formDataLayout);
	TextId = new QLineEdit;
	TextName = new QLineEdit;
	TextType = new QLineEdit;
	TextProducer = new QLineEdit;
	TextPrice = new QLineEdit;
	QLabel* idLabel = new QLabel("Id:");
	QLabel* nameLabel = new QLabel("Name:");
	QLabel* typeLabel = new QLabel("Type:");
	QLabel* producerLabel = new QLabel("Producer:");
	QLabel* priceLabel = new QLabel("Price:");
	//idLabel->setStyleSheet("color: white;");
	//nameLabel->setStyleSheet("color: white;");
	//typeLabel->setStyleSheet("color: white;");
	//producerLabel->setStyleSheet("color: white;");
	//priceLabel->setStyleSheet("color: white;");
	formDataLayout->addRow(idLabel, TextId);
	formDataLayout->addRow(nameLabel, TextName);
	formDataLayout->addRow(typeLabel, TextType);
	formDataLayout->addRow(producerLabel, TextProducer);
	formDataLayout->addRow(priceLabel, TextPrice);

	BtnAdd = new QPushButton("Add product");
	BtnUpdate = new QPushButton("Update product");
	BtnDelete = new QPushButton("Delete product");
	BtnUndo = new QPushButton("Undo");
	BtnBucket = new QPushButton("Open bucket");

	formDataLayout->addRow(BtnAdd);
	formDataLayout->addRow(BtnUpdate);
	formDataLayout->addRow(BtnDelete);
	formDataLayout->addRow(BtnUndo);
	formDataLayout->addRow(BtnBucket);
	formDataLayout->addRow(SecondWindow.BtnAddToBucket);
	formDataLayout->addRow(SecondWindow.BtnClear);
	formDataLayout->addRow(SecondWindow.BtnGenerate, SecondWindow.TextGenerate);
	formDataLayout->addRow(SecondWindow.BtnExport);

	MainLayout->addWidget(printingWidget);
	MainLayout->addWidget(formDataWidget);
	MainLayout->addWidget(&SecondWindow);
	SecondWindow.SetVisible(false);
}

void ProductsGUI::ConnectSignals()
{
	QObject::connect(
		BtnSortByName,
		&QPushButton::clicked,
		[&]() { ReloadList(Serv.Sort([](const Product& Product1, const Product& Product2) {return (Product1.GetName() < Product2.GetName()); })); }
	);
	QObject::connect(
		BtnSortByPrice,
		&QPushButton::clicked,
		[&]() { ReloadList(Serv.Sort([](const Product& Product1, const Product& Product2) {return (Product1.GetPrice() < Product2.GetPrice()); })); }
	);
	QObject::connect(
		BtnSortByNameAndProducer,
		&QPushButton::clicked,
		[&]() { ReloadList(Serv.Sort([](const Product& Product1, const Product& Product2)
			{
				if (Product1.GetName() != Product2.GetName())
				{
					return (Product1.GetName() < Product2.GetName());
				}
				return (Product1.GetType() < Product2.GetType());
			}));
		}
	);
	
	QObject::connect(
		TextFilterName,
		&QLineEdit::textChanged,
		[&]() {
			try
			{
				ReloadList(Serv.Filter(TextFilterName->text().toStdString(), 1));
			}
			catch (ValidationError& error)
			{
				ReloadList(Serv.GetAll());
			}
		}
	);
	QObject::connect(
		TextFilterProducer,
		&QLineEdit::textChanged,
		[&]() {
			try
			{
				ReloadList(Serv.Filter(TextFilterProducer->text().toStdString(), 2));
			}
			catch (ValidationError& error)
			{
				ReloadList(Serv.GetAll());
			}
		}
	);
	QObject::connect(
		TextFilterPrice,
		&QLineEdit::textChanged,
		[&]() {
			try
			{
				ReloadList(Serv.Filter(TextFilterPrice->text().toInt()));
			}
			catch (ValidationError& error)
			{
				ReloadList(Serv.GetAll());
			}
		}
	);
	
	QObject::connect(
		List,
		&QListWidget::itemSelectionChanged,
		[&]() {
			auto selected = List->selectedItems();
			if (selected.isEmpty())
			{
				TextId->setText("");
				TextName->setText("");
				TextType->setText("");
				TextProducer->setText("");
				TextPrice->setText("");
			}
			else
			{
				auto selectedItem = selected[0];
				Product toDisplay = Serv.FindProduct(selectedItem->data(Qt::UserRole).toInt());
				TextId->setText(QString::number(toDisplay.GetId()));
				TextName->setText(QString::fromStdString(toDisplay.GetName()));
				TextType->setText(QString::fromStdString(toDisplay.GetType()));
				TextProducer->setText(QString::fromStdString(toDisplay.GetProducer()));
				TextPrice->setText(QString::number(toDisplay.GetPrice()));
			}
		}
	);

	QObject::connect(
		BtnBucket, 
		&QPushButton::clicked, 
		[&]() { 
			if (SecondWindow.isVisible())
			{
				SecondWindow.SetVisible(false);
				BtnBucket->setText("Open bucket");
			}
			else
			{
				SecondWindow.SetVisible(true);
				BtnBucket->setText("Close bucket");
			}
		}
	);

	QObject::connect(BtnAdd, &QPushButton::clicked, this, &ProductsGUI::AddProduct);
	QObject::connect(BtnUpdate, &QPushButton::clicked, this, &ProductsGUI::UpdateProduct);
	QObject::connect(BtnDelete, &QPushButton::clicked, this, &ProductsGUI::DeleteProduct);
	QObject::connect(BtnUndo, &QPushButton::clicked, this, &ProductsGUI::Undo);
}

void ProductsGUI::ReloadList(const std::vector<Product>& Products)
{
	List->clear();
	for (const auto& product : Products)
	{
		QListWidgetItem* item = new QListWidgetItem(QString::fromStdString(product.GetName()));
		item->setData(Qt::UserRole, product.GetId());
		List->addItem(item);
	}
	if (List->count() == 0)
	{
		BtnBucket->setVisible(false);
	}
	else
	{
		BtnBucket->setVisible(true);
	}
}

void ProductsGUI::AddProduct()
{
	try
	{
		Serv.AddProduct(
			TextId->text().toInt(),
			TextName->text().toStdString(),
			TextType->text().toStdString(),
			TextProducer->text().toStdString(),
			TextPrice->text().toInt()
		);
		ReloadList(Serv.GetAll());
	}
	catch (ValidationError& error)
	{
		QMessageBox::warning(this, "Warning", QString::fromStdString(error.getMessage()));
	}
	catch (RepositoryError& error)
	{
		QMessageBox::warning(this, "Warning", QString::fromStdString(error.getMessage()));
	}
}

void ProductsGUI::UpdateProduct()
{
	try
	{
		Serv.ModifyProduct(
			TextId->text().toInt(),
			TextName->text().toStdString(),
			TextType->text().toStdString(),
			TextProducer->text().toStdString(),
			TextPrice->text().toInt()
		);
		ReloadList(Serv.GetAll());
	}
	catch (ValidationError& error)
	{
		QMessageBox::warning(this, "Warning", QString::fromStdString(error.getMessage()));
	}
	catch (RepositoryError& error)
	{
		QMessageBox::warning(this, "Warning", QString::fromStdString(error.getMessage()));
	}
}

void ProductsGUI::DeleteProduct()
{
	try
	{
		Serv.DeleteProduct(TextId->text().toInt());
		ReloadList(Serv.GetAll());
	}
	catch (ValidationError& error)
	{
		QMessageBox::warning(this, "Warning", QString::fromStdString(error.getMessage()));
	}
	catch (RepositoryError& error)
	{
		QMessageBox::warning(this, "Warning", QString::fromStdString(error.getMessage()));
	}
}

void ProductsGUI::Undo()
{
	try
	{
		Serv.UndoServ();
		ReloadList(Serv.GetAll());
	}
	catch (const GeneralExceptions& error)
	{
		QMessageBox::warning(this, "Warning", QString::fromStdString(error.getMessage()));
	}
}

void BucketGUI::SetVisible(bool V)
{
	this->setVisible(V);
	BtnAddToBucket->setVisible(V);
	BtnClear->setVisible(V);
	BtnGenerate->setVisible(V);
	if (false == V || BucketList->count() == 0)
	{
		BtnExport->setVisible(false);
	}
	else
	{
		BtnExport->setVisible(true);
	}
	TextGenerate->setVisible(V);
}

void BucketGUI::Init()
{
	Layout = new QHBoxLayout;
	setLayout(Layout);

	BucketList = new QListWidget;

	BtnAddToBucket = new QPushButton("Add product to bucket");
	BtnClear = new QPushButton("Clear bucket");
	BtnGenerate = new QPushButton("Generate");
	BtnExport = new QPushButton("Export bucket");

	TextGenerate = new QLineEdit;

	QLabel* idLabel = new QLabel("Id:");
	QLabel* nameLabel = new QLabel("Name:");
	QLabel* typeLabel = new QLabel("Type:");
	QLabel* producerLabel = new QLabel("Producer:");
	QLabel* priceLabel = new QLabel("Price:");
	QLabel* bucketValueLabel = new QLabel("Bucket value:");

	IdToShow = new QLabel;
	NameToShow = new QLabel;
	TypeToShow = new QLabel;
	ProducerToShow = new QLabel;
	PriceToShow = new QLabel;
	BucketValue = new QLabel("0");

	QWidget* printingWidget = new QWidget;
	QFormLayout* printingLayout = new QFormLayout;
	printingWidget->setLayout(printingLayout);

	printingLayout->addRow(BucketList);
	printingLayout->addRow(bucketValueLabel, BucketValue);
	printingLayout->addRow(idLabel, IdToShow);
	printingLayout->addRow(nameLabel, NameToShow);
	printingLayout->addRow(typeLabel, TypeToShow);
	printingLayout->addRow(producerLabel, ProducerToShow);
	printingLayout->addRow(priceLabel, PriceToShow);

	Layout->addWidget(printingWidget);
}

void BucketGUI::ConnectSignals()
{
	QObject::connect(
		BucketList,
		&QListWidget::itemSelectionChanged,
		[&]() {
			auto selected = BucketList->selectedItems();
			if (selected.isEmpty())
			{
				IdToShow->setText("");
				NameToShow->setText("");
				TypeToShow->setText("");
				ProducerToShow->setText("");
				PriceToShow->setText("");
			}
			else
			{
				auto selectedItem = selected[0];
				Product toDisplay = Serv.FindProduct(selectedItem->data(Qt::UserRole).toInt());
				IdToShow->setText(QString::number(toDisplay.GetId()));
				NameToShow->setText(QString::fromStdString(toDisplay.GetName()));
				TypeToShow->setText(QString::fromStdString(toDisplay.GetType()));
				ProducerToShow->setText(QString::fromStdString(toDisplay.GetProducer()));
				PriceToShow->setText(QString::number(toDisplay.GetPrice()));
			}
		}
	);

	QObject::connect(BtnAddToBucket, &QPushButton::clicked, this, &BucketGUI::AddBucket);
	QObject::connect(BtnClear, &QPushButton::clicked, this, &BucketGUI::Clear);
	QObject::connect(BtnGenerate, &QPushButton::clicked, this, &BucketGUI::Generate);
	QObject::connect(BtnExport, &QPushButton::clicked, this, &BucketGUI::Export);
}

void BucketGUI::ReloadList(const std::vector<Product>& BucketProducts)
{
	int value = 0;
	BucketList->clear();
	for (const auto& product : BucketProducts)
	{
		QListWidgetItem* item = new QListWidgetItem(QString::fromStdString(product.GetName()));
		item->setData(Qt::UserRole, product.GetId());
		BucketList->addItem(item);
		value = value + product.GetPrice();
	}
	BucketValue->setText(QString::number(value));
	if (BucketList->count() == 0)
	{
		BtnExport->setVisible(false);
	}
	else
	{
		BtnExport->setVisible(true);
	}
}

void BucketGUI::AddBucket()
{
	auto selected = ProductsList->selectedItems();
	if (selected.isEmpty() == false)
	{
		auto selectedItem = selected[0];
		ServBucket.addToBucket(selectedItem->data(Qt::UserRole).toInt());
	}
	ReloadList(ServBucket.getBucket());
}

void BucketGUI::Clear()
{
	ServBucket.clearBucket();
	ReloadList(ServBucket.getBucket());
}

void BucketGUI::Generate()
{
	ServBucket.generateBucket(TextGenerate->text().toInt());
	ReloadList(ServBucket.getBucket());
}

void BucketGUI::Export()
{
	std::vector < Product > products = ServBucket.getBucket();
	QString fileName = QFileDialog::getSaveFileName(this, "Export Bucket", "", "Comma-separated values (*.csv);;All Files (*)");

	if (fileName.isEmpty())
	{
		return;
	}

	QFile file(fileName);
	if (false == file.open(QIODevice::WriteOnly))
	{
		QMessageBox::information(this, "Unable to open file", file.errorString());
		return;
	}

	QTextStream out(&file);
	for (const auto& toExport : products)
	{
		out << QString::number(toExport.GetId()) << ","
			<< QString::fromStdString(toExport.GetName()) << ","
			<< QString::fromStdString(toExport.GetType()) << ","
			<< QString::fromStdString(toExport.GetProducer()) << ","
			<< QString::number(toExport.GetPrice()) << "\n";
	}
}