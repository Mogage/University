module ati {
    requires javafx.controls;
    requires javafx.fxml;

    requires org.controlsfx.controls;
    requires com.dlsc.formsfx;
    requires org.kordamp.bootstrapfx.core;
    requires java.sql;

    opens ati to javafx.fxml;
    exports ati;
    exports ati.controllers;
    exports ati.service;
    exports ati.repository;
    exports ati.domain;
    opens ati.controllers to javafx.fxml;
}