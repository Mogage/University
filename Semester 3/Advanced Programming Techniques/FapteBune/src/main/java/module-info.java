module faptebune {
    requires javafx.controls;
    requires javafx.fxml;

    requires org.controlsfx.controls;
    requires com.dlsc.formsfx;
    requires org.kordamp.bootstrapfx.core;
    requires java.sql;

    opens faptebune to javafx.fxml;
    exports faptebune;
    exports faptebune.domain;
    exports faptebune.service;
    exports faptebune.repository;
    exports faptebune.controllers;
    exports faptebune.utils;
    opens faptebune.controllers to javafx.fxml;
    opens faptebune.domain;
}