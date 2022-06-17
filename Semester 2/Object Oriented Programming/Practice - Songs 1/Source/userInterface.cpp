#include "userInterface.h"
#include "exceptions.h"
#include <qformlayout.h>
#include <qmessagebox.h>

void UserInterface::init()
{
	MainLayout = new QVBoxLayout;
	setLayout(MainLayout);

	Model = new TableModel{ Serv.all() };
	Table = new QTableView;
	Table->setModel(Model);

	QWidget* fieldsWidget = new QWidget;
	QFormLayout* fieldsLayout = new QFormLayout;
	fieldsWidget->setLayout(fieldsLayout);
	QLabel* titleLabel = new QLabel("Titlu: ");
	TextTitle = new QLineEdit;
	Slider = new QSlider(Qt::Horizontal);
	SliderValue = new QLabel("Val: 0");
	Slider->setTickInterval(1);
	Slider->setMaximum(10);
	BtnUpdate = new QPushButton("Update song");
	BtnDelete = new QPushButton("Delete song");

	fieldsLayout->addRow(titleLabel, TextTitle);
	fieldsLayout->addRow(SliderValue, Slider);
	fieldsLayout->addRow(BtnUpdate);
	fieldsLayout->addRow(BtnDelete);

	QWidget* topWidget = new QWidget;
	QHBoxLayout* topLayout = new QHBoxLayout;
	topWidget->setLayout(topLayout);
	topLayout->addWidget(Table);
	topLayout->addWidget(fieldsWidget);

	QHBoxLayout* DrawingLayout = new QHBoxLayout;
	Drawing = new QWidget;
	Drawing->setLayout(DrawingLayout);
	Drawing->setMinimumHeight(120);

	setMinimumHeight(300);
	setMinimumWidth(700);
	MainLayout->addWidget(topWidget);
	MainLayout->addWidget(Drawing);
}

void UserInterface::reset()
{
	TextTitle->setText("");
	Slider->setValue(0);
	SliderValue->setText("Val: 0");
	LastId = -1;
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
				reset();
				return;
			}
			int row = Table->selectionModel()->selectedIndexes().at(0).row();
			auto cell0Index = Table->model()->index(row, 0);
			int cell0Value = Table->model()->data(cell0Index, Qt::DisplayRole).toInt();
			Song song = Serv.find(cell0Value);
			LastId = cell0Value;
			TextTitle->setText(QString::fromStdString(song.title()));
			Slider->setValue(song.rank());
			SliderValue->setText("Val: " + QString::number(song.rank()));
		}
	);

	QObject::connect(
		Slider,
		&QSlider::valueChanged,
		[&]()
		{
			SliderValue->setText("Val: " + QString::number(Slider->value()));
		}
	);

	QObject::connect(BtnUpdate, &QPushButton::clicked, this, &UserInterface::Update);
	QObject::connect(BtnDelete, &QPushButton::clicked, this, &UserInterface::Delete);
}

void UserInterface::reloadList(vector<Song> Songs)
{
	Model->setSongs(Songs);
	repaint();
}

void UserInterface::Update()
{
	Serv.update(LastId, TextTitle->text().toStdString());
	Serv.update(LastId, Slider->value());
	reloadList(Serv.all());
}

void UserInterface::Delete()
{
	try
	{
		Serv.deleteId(LastId);
		reloadList(Serv.all());
		reset();
	}
	catch (const RepositoryError& error)
	{
		QMessageBox::warning(this, "Warning", QString::fromStdString(error.getMessage()));
	}
	catch (const ValidationError& error)
	{
		QMessageBox::warning(this, "Warning", QString::fromStdString(error.getMessage()));
	}
}

UserInterface::UserInterface(Service& _Serv) : Serv{ _Serv }
{
	init();
	connectSignals();
	reloadList(Serv.all());
}
