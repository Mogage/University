package anar.controllers;

import anar.domain.Rau;
import anar.service.Service;
import anar.utils.Observer;
import javafx.collections.FXCollections;
import javafx.collections.ObservableList;
import javafx.fxml.FXML;
import javafx.scene.control.*;
import javafx.scene.control.cell.PropertyValueFactory;

import java.util.ArrayList;
import java.util.List;

public class RauriController implements Observer {
    @FXML
    public TableView<Rau> rauriTable;
    @FXML
    public TableColumn<Rau, String> numeRauTableColumn;
    @FXML
    public TableColumn<Rau, Integer> cotaRauTableColumn;
    @FXML
    public TextField cotaTextField;
    @FXML
    public Button schimbaCotaButton;

    private Service service;

    private final ObservableList<Rau> rauriList = FXCollections.observableArrayList();

    public void initialise(Service service) {
        this.service = service;

        initRauri();
    }

    private void initRauri() {
        numeRauTableColumn.setCellValueFactory(new PropertyValueFactory<>("nume"));
        cotaRauTableColumn.setCellValueFactory(new PropertyValueFactory<>("cotaMedie"));
        updateRauri();
    }

    private void updateRauri() {
        Iterable<Rau> rauri = service.getRauri();
        List<Rau> auxRauri = new ArrayList<>();

        for(Rau rau : rauri) {
            auxRauri.add(rau);
        }

        rauriList.setAll(auxRauri);
        rauriTable.setItems(rauriList);
    }

    public void schimbaCota() {
        String cotaNoua = cotaTextField.getText();
        Rau rau = rauriTable.getSelectionModel().getSelectedItem();
        if (rau == null){
            Alert alert = new Alert(Alert.AlertType.ERROR, "Selecteaza un rau sa modifici cota", ButtonType.OK);
            alert.show();
            return;
        }
        if (cotaNoua.isEmpty()) {
            Alert alert = new Alert(Alert.AlertType.ERROR, "Cota noua nu poate fi vida", ButtonType.OK);
            alert.show();
            return;
        }
        try{
            Integer cota = Integer.parseInt(cotaNoua);
            service.updateRau(rau, cota);
        } catch (Exception e){
            Alert alert = new Alert(Alert.AlertType.ERROR, "Cota trebuie sa fie numar", ButtonType.OK);
            alert.show();
        }
    }

    @Override
    public void update() {
        updateRauri();
    }
}