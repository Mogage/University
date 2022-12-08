module com.socialNetwork {
    requires javafx.controls;
    requires javafx.fxml;

    requires org.controlsfx.controls;
    requires com.dlsc.formsfx;
    requires org.kordamp.bootstrapfx.core;
    requires java.sql;

    opens com.socialNetwork to javafx.fxml;
    exports com.socialNetwork;
    exports com.socialNetwork.controllers;
    exports com.socialNetwork.domain;
    exports com.socialNetwork.service;
    exports com.socialNetwork.repository;
    exports com.socialNetwork.exceptions;
    exports com.socialNetwork.network;
    exports com.socialNetwork.domain.validators;
    opens com.socialNetwork.controllers to javafx.fxml;
    opens com.socialNetwork.domain to javafx.base;
}