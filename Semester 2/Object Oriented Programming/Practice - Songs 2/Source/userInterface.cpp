#include "userInterface.h"
#include <qlabel.h>
#include <qformlayout.h>

void UserInterface::init()
{
	MainLayout = new QHBoxLayout;
	setLayout(MainLayout);

	Model = new TableModel{ Serv.all() };
	Table = new QTableView;
	Table->setModel(Model);

	QWidget* rightWidget = new QWidget;
	QFormLayout* rightLayout = new QFormLayout;
	QLabel* titleLabel = new QLabel("Title: ");
	QLabel* artistLabel = new QLabel("Artist: ");
	QLabel* genLabel = new QLabel("Gen: ");
	rightWidget->setLayout(rightLayout);
	TextTitle = new QLineEdit;
	TextArtist = new QLineEdit;
	TextGen = new QLineEdit;
	BtnAdd = new QPushButton("Add song");
	BtnDel = new QPushButton("Delete song");
	rightLayout->addRow(titleLabel, TextTitle);
	rightLayout->addRow(artistLabel, TextArtist);
	rightLayout->addRow(genLabel, TextGen);
	rightLayout->addRow(BtnAdd);
	rightLayout->addRow(BtnDel);

	MainLayout->addWidget(Table);
	MainLayout->addWidget(rightWidget);
	MainLayout->setContentsMargins(50, 50, 50, 50);
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
				LastId = -1;
				return;
			}
			int row = Table->selectionModel()->selectedIndexes().at(0).row();
			auto cell0Index = Table->model()->index(row, 0);
			LastId = Table->model()->data(cell0Index, Qt::DisplayRole).toInt();
		}
	);

	QObject::connect(BtnAdd, &QPushButton::clicked, this, &UserInterface::add);
	QObject::connect(BtnDel, &QPushButton::clicked, this, &UserInterface::del);
}

void UserInterface::reloadList(const vector<Song>& Songs)
{
	Model->setSongs(Songs);
	repaint();
}

void UserInterface::add()
{
	Serv.add(
		TextTitle->text().toStdString(),
		TextArtist->text().toStdString(),
		TextGen->text().toStdString()
	);
	reloadList(Serv.all());
}

void UserInterface::del()
{
	Serv.del(LastId);
	LastId = -1;
	reloadList(Serv.all());
}

UserInterface::UserInterface(Service& _Serv) : Serv{ _Serv }
{
	init();
	connectSignals();
	reloadList(Serv.all());
}
