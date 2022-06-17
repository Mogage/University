#pragma once
#include "service.h"
#include "models.h"
#include <qwidget.h>
#include <qlayout.h>
#include <qpushbutton.h>
#include <qlineedit.h>
#include <qslider.h>
#include <qlabel.h>
#include <qpainter.h>

class UserInterface :
    public QWidget
{
private:
    Service& Serv;
    int LastId = -1;

    QWidget* Drawing;
    QVBoxLayout* MainLayout;

    TableModel* Model;
    QTableView* Table;

    QPushButton* BtnUpdate;
    QPushButton* BtnDelete;

    QLineEdit* TextTitle;

    QSlider* Slider;
    QLabel* SliderValue;

    void init();
    void connectSignals();
    void reloadList(vector < Song > Songs);

    void reset();
    void Update();
    void Delete();
public:
    UserInterface(Service& _Serv);

    void paintEvent(QPaintEvent* ev) override {
        QPainter p{ this };
        vector < Song > songs = Serv.all();
        int distance = (width() - 100) / 11;
        int x = 50, y = this->height() - 20, max = 0;
        int ranks[11] = { 0 };
        for (auto& song : songs) {
            ranks[song.rank()]++;
            if (ranks[song.rank()] > max)
                max = ranks[song.rank()];
        }
        for (int i = 0; i < 11; i++) {
            p.setBrush(Qt::black);
            p.drawRect(x, y, 30, (double)ranks[i] / max * -100);
            p.drawText(x + 12, y + 15, QString::number(i));
            x += distance;
        }
    }
};

