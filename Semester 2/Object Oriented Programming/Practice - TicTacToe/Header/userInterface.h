#pragma once

#include "service.h"
#include "models.h"
#include <qwidget.h>
#include <qtableview.h>
#include <qlayout.h>
#include <qlineedit.h>
#include <qpushbutton.h>
#include <qcheckbox.h>
#include <qgridlayout.h>

class UserInterface :
    public QWidget
{
    Q_OBJECT
private:
    Service& Serv;
    int gameId = -1;
    
    TableModel* Model;
    QTableView* Table;

    QWidget* PlayGrid;
    QGridLayout* PlayGridLayout;

    QHBoxLayout* MainLayout;

    vector < vector < QPushButton* > > ButtonsForPlay;
    QPushButton* BtnCreateGame;
    QPushButton* BtnPlay;

    QLineEdit* TextDimensiune;
    QLineEdit* TextTabla;
    QLineEdit* TextFirst;

    void init();
    void connectSignals();
    void reloadList(vector < Game >& Games);

    void createGame();
public:
    UserInterface(Service& _Serv) : Serv{ _Serv } 
    {
        init();
        connectSignals();
        reloadList(Serv.all());
    }
};

