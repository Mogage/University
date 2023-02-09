module examen {
    requires javafx.controls;
    requires javafx.fxml;

    requires org.controlsfx.controls;
    requires com.dlsc.formsfx;
    requires org.kordamp.bootstrapfx.core;
    requires java.sql;

    opens examen to javafx.fxml;
    exports examen;
    exports examen.controllers;
    exports examen.domain;
    exports examen.repository;
    exports examen.service;
    opens examen.controllers to javafx.fxml;
}