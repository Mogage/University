module bms.bugmanagementsystem {
    requires javafx.controls;
    requires javafx.fxml;

    requires org.controlsfx.controls;
    requires com.dlsc.formsfx;
    requires org.kordamp.bootstrapfx.core;

    opens bms.bugmanagementsystem to javafx.fxml;
    exports bms.bugmanagementsystem;
}