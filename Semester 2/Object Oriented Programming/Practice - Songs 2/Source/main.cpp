#include <QtWidgets/QApplication>
#include "userInterface.h"

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    
    Repository repo;
    Service serv{ repo };
    UserInterface ui{ serv };

    ui.show();

    return a.exec();
}
