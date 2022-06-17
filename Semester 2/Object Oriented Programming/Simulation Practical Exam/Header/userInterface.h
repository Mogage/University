#pragma once

#include "service.h"

#include <qwidget.h>
#include <qlayout.h>
#include <qlistwidget.h>
#include <qpushbutton.h>
#include <qlineedit.h>
#include <map>

class UserInterface : public QWidget
{
	Q_OBJECT
private:
	Service& Serv;
	std::map < string, int > Evidence;
		
	QHBoxLayout* MainLayout;
	QVBoxLayout* LeftInfoLayout;

	QListWidget* List;

	QWidget* LeftInfoWidget;

	QLineEdit* TextCod;
	QLineEdit* TextBrand;
	QLineEdit* TextModel;
	QLineEdit* TextPret;

	QPushButton* ButtonSortPrice;
	QPushButton* ButtonSortBrand;
	QPushButton* ButtonSortModel;
	QPushButton* ButtonNotSorted;

	/// <summary>
	///		Initializeaza toate variabilele si le aranjeaza
	/// </summary>
	void Init();

	/// <summary>
	///		Creeaza conexiunile pentru butoane
	/// </summary>
	void ConnectSignals();

	/// <summary>
	///		Functie de adaugare dinamic butoane
	/// </summary>
	/// <param name="Telefoane">vector ce contine telefoane</param>
	void AddButtons(const vector < Telefon >& Telefoane);

	/// <summary>
	///		Reincarca Lista de afisare conform unui vector cu telefoane
	/// </summary>
	/// <param name="Telefoane">vector ce contine telefoane</param>
	void ReloadList(const vector < Telefon >& Telefoane);
public:
	UserInterface(Service& _Serv) : Serv { _Serv }
	{
		Init();
		ConnectSignals();
		ReloadList(Serv.getAll());
	}
};

