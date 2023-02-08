package ati.controllers;

import ati.domain.Pacient;
import ati.service.Service;
import ati.utils.Observer;
import javafx.collections.FXCollections;
import javafx.collections.ObservableList;
import javafx.fxml.FXML;
import javafx.scene.control.*;
import javafx.scene.control.cell.PropertyValueFactory;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

public class AsteptariController{
    @FXML
    public TableView<Pacient> pacientiTable;
    @FXML
    public TableColumn<Pacient, String> cnpTableColumn;
    @FXML
    public TableColumn<Pacient, String> diagnosticTableColumn;
    @FXML
    public Button ticButton;
    @FXML
    public Button timButton;
    @FXML
    public Button tiipButton;

    private Service service;

    private final ObservableList<Pacient> pacientiList = FXCollections.observableArrayList();

    public void initialise(Service service) {
        this.service = service;

        initPacienti();
    }

    private void initPacienti() {
        cnpTableColumn.setCellValueFactory(new PropertyValueFactory<>("cnp"));
        diagnosticTableColumn.setCellValueFactory(new PropertyValueFactory<>("diagnostic"));
        updatePacienti();
    }

    private void updatePacienti() {
        Iterable<Pacient> pacienti = service.getPacientiAsteptare();
        List<Pacient> pacientList = new ArrayList<>();

        for (Pacient pacient : pacienti) {
            pacientList.add(pacient);
        }

        pacientiList.setAll(pacientList.stream().sorted((pacient1, pacient2) -> pacient2.getGravitate().compareTo(pacient1.getGravitate())).collect(Collectors.toList()));
        pacientiTable.setItems(pacientiList);
    }

    private void moveToBed(String tipPat) {
        Pacient pacient = pacientiTable.getSelectionModel().getSelectedItem();
        if (pacient == null) {
            Alert alert = new Alert(Alert.AlertType.ERROR, "Nu a fost ales niciun pacient.", ButtonType.OK);
            alert.show();
            return;
        }
        try {
            service.ocupaPat(tipPat, pacient.getCnp());
        } catch (Exception e) {
            Alert alert = new Alert(Alert.AlertType.ERROR, e.getMessage(), ButtonType.OK);
            alert.show();
            return;
        }
        updatePacienti();
    }

    public void moveToTicBed() {
        moveToBed("TIC");
    }

    public void moveToTimButton() {
        moveToBed("TIM");
    }

    public void moveToTiipButton() {
        moveToBed("TIIP");
    }
}
