#include <QtWidgets/QApplication>

#include "userInterface.h"

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    
    Repository repo;
    Validator valid;
    Service serv{ repo, valid };
    UserInterface ui{ serv };

    ui.show();

    return a.exec();
}
