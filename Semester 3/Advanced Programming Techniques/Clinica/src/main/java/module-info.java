module clinica {
    requires javafx.controls;
    requires javafx.fxml;

    requires org.controlsfx.controls;
    requires com.dlsc.formsfx;
    requires org.kordamp.bootstrapfx.core;
    requires java.sql;

    opens clinica to javafx.fxml;
    exports clinica;
    exports clinica.controllers;
    exports clinica.service;
    exports clinica.domain;
    exports clinica.repository;
    opens clinica.controllers to javafx.fxml;
}