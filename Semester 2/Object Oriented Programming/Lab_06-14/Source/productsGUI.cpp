#include "productsGUI.h"
#include "exceptions.h"
#include <qformlayout.h>
#include <qmessagebox.h>
#include <qgridlayout.h>
#include <qfiledialog.h>
#include <qheaderview.h>
#include <qitemselectionmodel.h>
#include <qdebug.h>
#include <qfile.h>
#include <map>

void ProductsGUI::Init()
{
	MainLayout = new QHBoxLayout;
	setLayout(MainLayout);
	// setStyleSheet("background-color: black;");

	QWidget* printingWidget = new QWidget;
	QVBoxLayout* printingLayout = new QVBoxLayout;
	printingWidget->setLayout(printingLayout);

	ListWid = new QListWidget;

	List = new QListView;
	model = new ListModel{ Serv.GetAll() };
	List->setUniformItemSizes(true);		
	List->setModel(model);

	//List = new QListWidget;
	/*List = new QTableWidget;
	List->verticalHeader()->setVisible(false);
	List->setColumnCount(5);
	List->setHorizontalHeaderItem(0, new QTableWidgetItem("Id"));
	List->setHorizontalHeaderItem(1, new QTableWidgetItem("Name"));
	List->setHorizontalHeaderItem(2, new QTableWidgetItem("Type"));
	List->setHorizontalHeaderItem(3, new QTableWidgetItem("Producer"));
	List->setHorizontalHeaderItem(4, new QTableWidgetItem("Price"));*/

	QWidget* gridButtonsWidget = new QWidget;
	QGridLayout* gridButtonsLayout = new QGridLayout;
	gridButtonsWidget->setLayout(gridButtonsLayout);
	BtnSortByName = new QPushButton("Sort by name");
	BtnSortByPrice = new QPushButton("Sort by price");
	BtnSortByNameAndProducer = new QPushButton("Sort by name and producer");
	TextFilterName = new QLineEdit;
	TextFilterProducer = new QLineEdit;
	TextFilterPrice = new QLineEdit;
	TextFilterName->setPlaceholderText("Filter by name");
	TextFilterProducer->setPlaceholderText("Filter by Producer");
	TextFilterPrice->setPlaceholderText("Filter by Price");
	gridButtonsLayout->addWidget(BtnSortByName, 0, 0);
	gridButtonsLayout->addWidget(BtnSortByPrice, 1, 0);
	gridButtonsLayout->addWidget(BtnSortByNameAndProducer, 2, 0);
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
	BtnOpenBucket = new QPushButton("Open bucket");
	BtnPaintBucket = new QPushButton("Open read only");
	BtnAddToBucket = new QPushButton("Add product to bucket");
	BtnClear = new QPushButton("Clear bucket");
	BtnGenerate = new QPushButton("Generate");
	BtnExport = new QPushButton("Export bucket");

	TextGenerate = new QLineEdit;

	formDataLayout->addRow(BtnAdd);
	formDataLayout->addRow(BtnUpdate);
	formDataLayout->addRow(BtnDelete);
	formDataLayout->addRow(BtnUndo);
	formDataLayout->addRow(BtnOpenBucket, BtnPaintBucket);
	formDataLayout->addRow(BtnAddToBucket);
	formDataLayout->addRow(BtnClear);
	formDataLayout->addRow(BtnGenerate, TextGenerate);
	formDataLayout->addRow(BtnExport);
	/*formDataLayout->addRow(SecondWindow.BtnAddToBucket);
	formDataLayout->addRow(SecondWindow.BtnClear);
	formDataLayout->addRow(SecondWindow.BtnGenerate, SecondWindow.TextGenerate);
	formDataLayout->addRow(SecondWindow.BtnExport);*/

	LeftInfoWidget = new QWidget;
	LeftInfoLayout = new QVBoxLayout;
	LeftInfoWidget->setLayout(LeftInfoLayout);

	MainLayout->addWidget(LeftInfoWidget);
	MainLayout->addWidget(printingWidget);
	MainLayout->addWidget(formDataWidget);
	// MainLayout->addWidget(&SecondWindow);
	// SecondWindow.SetVisible(false);
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
		List->selectionModel(),
		&QItemSelectionModel::selectionChanged,
		//&QListWidget::itemSelectionChanged,
		//&QTableWidget::itemSelectionChanged,
		[this]() {
			//auto selected = List->selectedItems();
			
			if (List->selectionModel()->selectedIndexes().isEmpty())
			{
				TextId->setText("");
				TextName->setText("");
				TextType->setText("");
				TextProducer->setText("");
				TextPrice->setText("");
			}
			else
			{
				/*auto selectedItem = selected[0];
				Product toDisplay = Serv.FindProduct(selectedItem->data(Qt::UserRole).toInt());
				TextId->setText(QString::number(toDisplay.GetId()));
				TextName->setText(QString::fromStdString(toDisplay.GetName()));
				TextType->setText(QString::fromStdString(toDisplay.GetType()));
				TextProducer->setText(QString::fromStdString(toDisplay.GetProducer()));
				TextPrice->setText(QString::number(toDisplay.GetPrice()));*/

				int row = List->selectionModel()->selectedIndexes().at(0).row();
				auto cell = List->model()->index(row, 0);
				int id = List->model()->data(cell, Qt::DisplayRole).toString().split(" ")[0].toInt();
				
				Product toDisplay = Serv.FindProduct(id);
				TextId->setText(QString::number(toDisplay.GetId()));
				TextName->setText(QString::fromStdString(toDisplay.GetName()));
				TextType->setText(QString::fromStdString(toDisplay.GetType()));
				TextProducer->setText(QString::fromStdString(toDisplay.GetProducer()));
				TextPrice->setText(QString::number(toDisplay.GetPrice())); 
			}
		}
	);

	QObject::connect(
		BtnOpenBucket, 
		&QPushButton::clicked, 
		this,
		&ProductsGUI::AddWindow
		//[&]() { 
		//	BucketGUI* toAdd = new BucketGUI{ Serv, ServBck, List };
		//	toAdd->show();
		//	SecondWindows.push_back(toAdd);
		//	// QObject::connect(toAdd, &QWidget::isVisible, [&]() { SecondWindows.erase(std::remove(SecondWindows.begin(), SecondWindows.end(), toAdd)); qDebug() << SecondWindows.size(); });
		//	//SecondWindow.SetVisible(true);
		//	/*if (SecondWindow.isVisible()) //bucket in main window app
		//	{
		//		SecondWindow.SetVisible(false);
		//		BtnBucket->setText("Open bucket");
		//	}
		//	else
		//	{
		//		SecondWindow.SetVisible(true);
		//		BtnBucket->setText("Close bucket");
		//	}*/
		//}
	);

	QObject::connect(BtnPaintBucket, &QPushButton::clicked, this, &ProductsGUI::AddPaintWindow);

	QObject::connect(BtnAdd, &QPushButton::clicked, this, &ProductsGUI::AddProduct);
	QObject::connect(BtnUpdate, &QPushButton::clicked, this, &ProductsGUI::UpdateProduct);
	QObject::connect(BtnDelete, &QPushButton::clicked, this, &ProductsGUI::DeleteProduct);
	QObject::connect(BtnUndo, &QPushButton::clicked, this, &ProductsGUI::Undo);
}

void ProductsGUI::ReloadList(const std::vector<Product>& Products)
{
	//List->setRowCount(0);
	//List->clear();
	model->setProducts(Products);
	/*for (const auto& product : Products)
	{
		QListWidgetItem* item = new QListWidgetItem(QString::fromStdString(product.GetName()));
		item->setData(Qt::UserRole, product.GetId());
		List->addItem(item);*/
		/*QTableWidgetItem* id = new QTableWidgetItem(QString::number(product.GetId()));
		id->setData(Qt::UserRole, product.GetId());
		QTableWidgetItem* name = new QTableWidgetItem(QString::fromStdString(product.GetName()));
		name->setData(Qt::UserRole, product.GetId());
		QTableWidgetItem* type = new QTableWidgetItem(QString::fromStdString(product.GetType()));
		type->setData(Qt::UserRole, product.GetId());
		QTableWidgetItem* producer = new QTableWidgetItem(QString::fromStdString(product.GetProducer()));
		producer->setData(Qt::UserRole, product.GetId());
		QTableWidgetItem* price = new QTableWidgetItem(QString::number(product.GetPrice()));
		price->setData(Qt::UserRole, product.GetId());
		
		List->insertRow( List->rowCount() );
		
		List->setItem(List->rowCount() - 1, 0, id);
		List->setItem(List->rowCount() - 1, 1, name);
		List->setItem(List->rowCount() - 1, 2, type);
		List->setItem(List->rowCount() - 1, 3, producer);
		List->setItem(List->rowCount() - 1, 4, price);*/
	//}
	/*if (List->count() == 0)
	//if(List->rowCount() == 0)
	{
		BtnOpenBucket->setVisible(false);
		//SecondWindow.SetVisible(false);
	}
	else
	{
		BtnOpenBucket->setVisible(true);
	}
	for (auto window : SecondWindows)
	{
		window->reload();
	}*/
	//SecondWindow.reload();
	AddButtons(Products);
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

void ProductsGUI::AddButtons(const std::vector<Product>& Products)
{
	evidence.clear();
	for (const auto& product : Products)
	{
		evidence[product.GetType()] = evidence[product.GetType()] + 1;
	}

	ClearLayout(LeftInfoLayout);
	for (const auto& type : evidence)
	{
		QPushButton* toAdd = new QPushButton(QString::fromStdString(type.first));
		LeftInfoLayout->addWidget(toAdd);
		QObject::connect(
			toAdd,
			&QPushButton::clicked,
			[&]() {
				QMessageBox::information(NULL, "Info", QString("Numar de produse de acest tip:") + QString::number(type.second));
			}
		);
	}
}

void ProductsGUI::AddWindow()
{
	BucketGUI* toAdd = new BucketGUI{ Serv, ServBck, List };
	SecondWindows.push_back(toAdd);
	
	toAdd->BtnClear = BtnClear;
	toAdd->BtnAddToBucket = BtnAddToBucket;
	toAdd->BtnGenerate = BtnGenerate;
	toAdd->BtnExport = BtnExport;
	toAdd->TextGenerate = TextGenerate;
	if (1 == SecondWindows.size())
	{
		toAdd->connect();
	}
	else
	{
		toAdd->show();
	}
	toAdd->setAttribute(Qt::WA_DeleteOnClose);
	QObject::connect(toAdd, &QWidget::destroyed, [&]() {
		for(auto it = SecondWindows.begin(); it != SecondWindows.end(); it++)
		{
			if (toAdd = *it)
			{
				if (it == SecondWindows.begin() && SecondWindows.size() > 1)
				{
					SecondWindows.erase(it);
					SecondWindows[0]->connect();
				}
				else
				{
					SecondWindows.erase(it);
				}
				break;
			}
		}
		});
}

void ProductsGUI::AddPaintWindow()
{
	PaintGUI* toAdd = new PaintGUI{ ServBck };
	PaintWindows.push_back(toAdd);
	toAdd->show();
	toAdd->setAttribute(Qt::WA_DeleteOnClose);
	QObject::connect(toAdd, &QWidget::destroyed, [&]() {
		for (auto it = PaintWindows.begin(); it != PaintWindows.end(); it++)
		{
			if (toAdd = *it)
			{
				PaintWindows.erase(it);
				break;
			}
		}
		});
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
	if (false == V || BucketList->model()->rowCount() == 0)
	//if (false == V || BucketList->count() == 0)
	//if (false == V || BucketList->rowCount() == 0)
	{
		BtnExport->setVisible(false);
	}
	else
	{
		BtnExport->setVisible(true);
	}
	TextGenerate->setVisible(V);
}

void BucketGUI::reload()
{
	ReloadList(ServBucket.getBucket());
}

void BucketGUI::connect()
{
	QObject::connect(BtnAddToBucket, &QPushButton::clicked, this, &BucketGUI::AddBucket);
	QObject::connect(BtnClear, &QPushButton::clicked, this, &BucketGUI::Clear);
	QObject::connect(BtnGenerate, &QPushButton::clicked, this, &BucketGUI::Generate);
	QObject::connect(BtnExport, &QPushButton::clicked, this, &BucketGUI::Export);
}

void BucketGUI::update()
{
	reload();
}

void BucketGUI::Init()
{
	Layout = new QHBoxLayout;
	setLayout(Layout);

	Model = new ListModel{ Serv.GetAll() };
	BucketList = new QListView;
	BucketList->setUniformItemSizes(true);
	BucketList->setModel(Model);
	/*BucketList = new QListWidget;
	BucketList = new QTableWidget;
	BucketList->setColumnCount(1);
	BucketList->setHorizontalHeaderItem(0, new QTableWidgetItem("Name"));
	BucketList->verticalHeader()->setVisible(false);*/

	//BtnAddToBucket = new QPushButton("Add product to bucket");
	//BtnClear = new QPushButton("Clear bucket");
	//BtnGenerate = new QPushButton("Generate");
	//BtnExport = new QPushButton("Export bucket");
	BtnClearInWindow = new QPushButton("Clear bucket");
	BtnGenerateInWindow = new QPushButton("Generate");

	//TextGenerate = new QLineEdit;
	TextGenerateInWindow = new QLineEdit;

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

	//printingLayout->addRow(BucketList);
	printingLayout->addRow(bucketValueLabel, BucketValue);
	printingLayout->addRow(idLabel, IdToShow);
	printingLayout->addRow(nameLabel, NameToShow);
	printingLayout->addRow(typeLabel, TypeToShow);
	printingLayout->addRow(producerLabel, ProducerToShow);
	printingLayout->addRow(priceLabel, PriceToShow);
	printingLayout->addRow(BtnClearInWindow);
	printingLayout->addRow(BtnGenerateInWindow, TextGenerateInWindow);

	Layout->addWidget(BucketList);
	Layout->addWidget(printingWidget);
}

void BucketGUI::ConnectSignals()
{

	ServBucket.addObserver(this);
	QObject::connect(
		BucketList->selectionModel(),
		&QItemSelectionModel::selectionChanged, 
		//&QListWidget::itemSelectionChanged,
		//&QTableWidget::itemSelectionChanged,
		[this]() {
			if (BucketList->selectionModel()->selectedRows().isEmpty())
			{
				IdToShow->setText("");
				NameToShow->setText("");
				TypeToShow->setText("");
				ProducerToShow->setText("");
				PriceToShow->setText("");
				return;
			}
			int row = BucketList->selectionModel()->selectedIndexes().at(0).row();
			auto cell = BucketList->model()->index(row, 0);
			int id = BucketList->model()->data(cell, Qt::DisplayRole).toString().split(" ")[0].toInt();

			Product toDisplay = Serv.FindProduct(id);
			IdToShow->setText(QString::number(toDisplay.GetId()));
			NameToShow->setText(QString::fromStdString(toDisplay.GetName()));
			TypeToShow->setText(QString::fromStdString(toDisplay.GetType()));
			ProducerToShow->setText(QString::fromStdString(toDisplay.GetProducer()));
			PriceToShow->setText(QString::number(toDisplay.GetPrice()));
			/*auto selected = BucketList->selectedItems();
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
			}*/
		}
	);

	//QObject::connect(BtnAddToBucket, &QPushButton::clicked, this, &BucketGUI::AddBucket);
	//QObject::connect(BtnClear, &QPushButton::clicked, this, &BucketGUI::Clear);
	QObject::connect(BtnClearInWindow, &QPushButton::clicked, this, &BucketGUI::Clear);
	//QObject::connect(BtnGenerate, &QPushButton::clicked, this, &BucketGUI::Generate);
	QObject::connect(BtnGenerateInWindow, &QPushButton::clicked, this, &BucketGUI::GenerateInWindow);
	//QObject::connect(BtnExport, &QPushButton::clicked, this, &BucketGUI::Export);
}

void BucketGUI::ReloadList(const std::vector<Product>& BucketProducts)
{
	int value = 0;
	Model->setProducts(BucketProducts);
	//BucketList->clear();
	//BucketList->setRowCount(0);
	for (const auto& product : BucketProducts)
	{
		Product toAdd = Serv.FindProduct(product.GetId());
		if (-1 == toAdd.GetId())
		{
			continue;
		}
		/*QListWidgetItem* item = new QListWidgetItem(QString::fromStdString(toAdd.GetName()));
		item->setData(Qt::UserRole, toAdd.GetId());
		BucketList->addItem(item);*/
		/*QTableWidgetItem* item = new QTableWidgetItem(QString::fromStdString(toAdd.GetName()));
		item->setData(Qt::UserRole, toAdd.GetId());
		
		BucketList->insertRow(BucketList->rowCount());
		BucketList->setItem(BucketList->rowCount() - 1, 0, item);*/
		value = value + toAdd.GetPrice();
	}
	BucketValue->setText(QString::number(value));
	//if (BucketList->count() == 0)
	////if (BucketList->rowCount() == 0)
	//{
	//	BtnExport->setVisible(false);
	//}
	//else
	//{
	//	BtnExport->setVisible(true);
	//}
}

void BucketGUI::AddBucket()
{
	if (false == ProductsList->selectionModel()->selectedIndexes().isEmpty())
	{
		int row = ProductsList->selectionModel()->selectedIndexes().at(0).row();
		auto cell = ProductsList->model()->index(row, 0);
		int id = ProductsList->model()->data(cell, Qt::DisplayRole).toString().split(" ")[0].toInt();
		ServBucket.addToBucket(id);
	}

	/*auto selected = ProductsList->selectedItems();
	if (selected.isEmpty() == false)
	{
		auto selectedItem = selected[0];
		ServBucket.addToBucket(selectedItem->data(Qt::UserRole).toInt());
	}*/
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

void BucketGUI::GenerateInWindow()
{
	ServBucket.generateBucket(TextGenerateInWindow->text().toInt());
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