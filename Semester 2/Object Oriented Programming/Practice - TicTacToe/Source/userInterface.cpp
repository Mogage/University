#include "userInterface.h"
#include "exceptions.h"

#include <qformlayout.h>
#include <qlabel.h>
#include <qmessagebox.h>

void UserInterface::init()
{
	MainLayout = new QHBoxLayout;
	setLayout(MainLayout);

	Model = new TableModel{ Serv.all() };
	Table = new QTableView;
	Table->setModel(Model);

	QWidget* formCreateGame = new QWidget;
	QFormLayout* formCreateGameLayout = new QFormLayout;
	formCreateGame->setLayout(formCreateGameLayout);
	TextDimensiune = new QLineEdit;
	TextTabla = new QLineEdit;
	TextFirst = new QLineEdit;
	QLabel* dimensiuneLabel = new QLabel{ "Dimensiune: " };
	QLabel* tablaLabel = new QLabel{ "Tabla: " };
	QLabel* firstLabel = new QLabel{ "Primul jucator: " };
	formCreateGameLayout->addRow(dimensiuneLabel, TextDimensiune);
	formCreateGameLayout->addRow(tablaLabel, TextTabla);
	formCreateGameLayout->addRow(firstLabel, TextFirst);

	BtnCreateGame = new QPushButton{ "Create Game" };
	formCreateGameLayout->addRow(BtnCreateGame);

	PlayGrid = new QWidget;
	PlayGridLayout = new QGridLayout;
	PlayGrid->setLayout(PlayGridLayout);

	QWidget* print = new QWidget;;
	QVBoxLayout* printLayout = new QVBoxLayout;
	print->setLayout(printLayout);
	printLayout->addWidget(Table);
	printLayout->addWidget(formCreateGame);

	MainLayout->addWidget(print);
	MainLayout->addWidget(PlayGrid);
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

void UserInterface::connectSignals()
{
	QObject::connect(
		Table->selectionModel(),
		&QItemSelectionModel::selectionChanged,
		[this]()
		{
			if (Table->selectionModel()->selectedIndexes().isEmpty())
			{
				TextDimensiune->setText("");
				TextTabla->setText("");
				TextFirst->setText("");
				gameId = -1;
				return;
			}
			int row = Table->selectionModel()->selectedIndexes().at(0).row();
			auto cell0Index = Table->model()->index(row, 0);
			int cell0Value = Table->model()->data(cell0Index, Qt::DisplayRole).toInt();
			Game joc = Serv.find(cell0Value);
			auto tabla = joc.tabla();
			gameId = cell0Value;
			TextDimensiune->setText(QString::number(joc.dim()));
			TextTabla->setText(QString::fromStdString(joc.tablaSir()));
			TextFirst->setText(QString(joc.curent()));
			for (auto it = ButtonsForPlay.begin(); it != ButtonsForPlay.end(); ++it)
				for (auto it2 = (*it).begin(); it2 != (*it).end(); ++it2)
					delete (*it2);
			ButtonsForPlay = vector < vector < QPushButton* > >(joc.dim(), vector < QPushButton*  >(joc.dim()));
			ClearLayout(PlayGridLayout);
			for (int row = 0; row < joc.dim(); ++row)
			{
				for (int col = 0; col < joc.dim(); ++col)
				{
					if ('-' == tabla[row][col])
					{
						ButtonsForPlay[row][col] = new QPushButton("");
					}
					else
					{
						ButtonsForPlay[row][col] = new QPushButton(QString(tabla[row][col]));
					}
					PlayGridLayout->addWidget(ButtonsForPlay[row][col], row, col);

					QObject::connect(
						ButtonsForPlay[row][col],
						&QPushButton::clicked,
						[this, row, col]()
						{
							char aux = Serv.move(gameId, row, col);
							reloadList(Serv.all());
							aux = (aux == '-') ? ' ' : aux;
							ButtonsForPlay[row][col]->setText(QString(aux));
						}
					);
				}
			}
		}
	);

	QObject::connect(BtnCreateGame, &QPushButton::clicked, this, &UserInterface::createGame);
}

void UserInterface::reloadList(vector<Game>& Games)
{
	Model->setGames(Games);
}

void UserInterface::createGame()
{
	try
	{
		Serv.createGame(
			TextDimensiune->text().toInt(),
			TextTabla->text().toStdString(),
			std::toupper(TextFirst->text().toStdString()[0])
		);
		reloadList(Serv.all());
	}
	catch (ValidationError& error)
	{
		QMessageBox::warning(this, "Warning", QString::fromStdString(error.getMessage()));
	}
}
