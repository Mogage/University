module root.proiect_mpp {
    requires javafx.controls;
    requires javafx.fxml;

    requires org.controlsfx.controls;
    requires com.dlsc.formsfx;
    requires org.kordamp.bootstrapfx.core;
    requires java.sql;
    requires org.apache.logging.log4j;

    opens root.proiect_mpp to javafx.fxml;
    exports root.proiect_mpp;
    exports root.proiect_mpp.controllers;
    opens root.proiect_mpp.controllers to javafx.fxml;
}