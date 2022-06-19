#pragma once

#include "service.h"
#include "models.h"
#include <qwidget.h>
#include <qlayout.h>
#include <qpushbutton.h>
#include <qlineedit.h>
#include <qpainter.h>
#include <map>

class UserInterface :
    public QWidget
{
private:
    Service& Serv;
    int LastId = -1;

    TableModel* Model;
    QTableView* Table;

    QHBoxLayout* MainLayout;

    QPushButton* BtnAdd;
    QPushButton* BtnDel;

    QLineEdit* TextTitle;
    QLineEdit* TextArtist;
    QLineEdit* TextGen;

    void init();
    void connectSignals();
    void reloadList(const vector < Song >& Songs);

    void add();
    void del();
public:
    UserInterface(Service& _Serv);

    void paintEvent(QPaintEvent* Event) override
    {
        QPainter painter{ this };
        vector < Song > songs = Serv.all();
        vector < int > raze( 4, 5 );
        vector < string > genuri = { "pop", "rock", "folk", "disco" };
        vector < QPoint > centers = { QPoint{25, 25}, QPoint{width() - 25, 25}, QPoint{25, height() - 25}, QPoint{width() - 25, height() - 25}};
        std::map < string, int > evidence;
        for (const auto& song : songs)
        {
            evidence[song.gen()]++;
        }
        for (int gen = 0; gen < 4; ++gen)
        {
            for (int cont = 0; cont < evidence[genuri[gen]]; ++cont)
            {
                painter.drawEllipse(centers[gen], raze[gen], raze[gen]);
                raze[gen] += 5;
            }
        }
    }
};

