module anar {
    requires javafx.controls;
    requires javafx.fxml;

    requires org.controlsfx.controls;
    requires com.dlsc.formsfx;
    requires org.kordamp.bootstrapfx.core;
    requires java.sql;

    opens anar to javafx.fxml;
    exports anar;
    exports anar.controllers;
    exports anar.domain;
    exports anar.service;
    exports anar.repository;
    opens anar.controllers to javafx.fxml;
}