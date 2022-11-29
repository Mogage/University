module com.example.gui_retea {
    requires javafx.controls;
    requires javafx.fxml;

    requires org.controlsfx.controls;
    requires com.dlsc.formsfx;
    requires org.kordamp.bootstrapfx.core;

    opens com.example.gui_retea to javafx.fxml;
    exports com.example.gui_retea;
}