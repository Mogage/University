#pragma once

#include "service.h"
#include "observer.h"
#include "models.h"
#include <map>
#include <qwidget.h>
#include <qlayout.h>
#include <qlineedit.h>
#include <qpushbutton.h>
#include <qslider.h>
#include <qlabel.h>

class UserInterface : public QWidget, public Observable
{
	Q_OBJECT
	friend class NumberType;
private:
	Service& Serv;
	std::map<string, int > Evidence;

	QHBoxLayout* MainLayout;
	TableModel* Model;
	QTableView* Table;

	QPushButton* BtnAdd;
	QSlider* Slider;
	QLabel* SliderValue;

	QLineEdit* TextId;
	QLineEdit* TextName;
	QLineEdit* TextType;
	QLineEdit* TextPrice;

	void init();
	void connectSignals();
	void reloadList(vector < Produs > Products);
	void createWindows();

	void add();
public:
	UserInterface(Service& _Serv);
};

class NumberType : public QWidget, public Observer
{
private:
	UserInterface& Ui;

	QHBoxLayout* MainLayout;
	QLabel* Number;
public:
	NumberType(UserInterface& _Ui,string Name, int _Number);
	void update() override;
};