#include "userInterface.h"
#include <QtWidgets/QApplication>

#include "userInterface.h"
#include "test.h"

int main(int argc, char* argv[])
{
    QApplication a(argc, argv);

    Test teste;

    teste.run();

    FileRepository repo;
    Service serv{ repo };
    UserInterface ui{ serv };

    ui.show();

    return a.exec();
}
