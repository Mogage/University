#include "test.h"
#include "productsGUI.h"
#include "repository.h"
#include "validator.h"
#include "service.h"
#include <qapplication.h>

int main(int argc, char *argv[])
{
    QApplication app(argc, argv);
    
    Test test;
    FileRepository repo{ "out.csv" };
    Bucket bucket;
    Validator valid;
    Service serv{ repo, valid };
    ServiceBucket servBck{repo, bucket, valid};
    ProductsGUI ui{ serv, servBck };

    test.RunAll();

    ui.show();
    ui.activateWindow();

    return app.exec();
}
