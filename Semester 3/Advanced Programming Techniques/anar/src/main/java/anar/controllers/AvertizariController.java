package anar.controllers;

import anar.domain.Localitate;
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
import java.util.stream.Collectors;

public class AvertizariController implements Observer {
    @FXML
    public TableView<Localitate> redusTable;
    @FXML
    public TableView<Localitate> mediuTable;
    @FXML
    public TableView<Localitate> majorTable;
    @FXML
    public TableColumn<Localitate, String> redusColumn;
    @FXML
    public TableColumn<Localitate, String> mediuColumn;
    @FXML
    public TableColumn<Localitate, String> majorColumn;
    @FXML
    public TextField rauTextField;
    @FXML
    public TextField cmdrTextField;
    @FXML
    public TextField cmaTextField;
    @FXML
    public Button riscButton;
    @FXML
    public Label riscLabel;

    private Service service;

    private final ObservableList<Localitate> redusList = FXCollections.observableArrayList();
    private final ObservableList<Localitate> mediuList = FXCollections.observableArrayList();
    private final ObservableList<Localitate> majorList = FXCollections.observableArrayList();

    public void initialise(Service service) {
        this.service = service;

        initAvertizari();
    }

    private void initAvertizari() {
        redusColumn.setCellValueFactory(new PropertyValueFactory<>("nume"));
        mediuColumn.setCellValueFactory(new PropertyValueFactory<>("nume"));
        majorColumn.setCellValueFactory(new PropertyValueFactory<>("nume"));

        updateAvertizari();
    }

    private void updateAvertizari() {
        Iterable<Localitate> localitati = service.getLocalitati();
        List<Localitate> auxLocalitati = new ArrayList<>();

        for (Localitate localitate : localitati) {
            auxLocalitati.add(localitate);
        }

        redusList.setAll(auxLocalitati.stream().filter((localitate) -> localitate.getCotaMinimaDeRisc() > localitate.getRau().getCotaMedie()).collect(Collectors.toList()));
        mediuList.setAll(auxLocalitati.stream().filter((localitate) -> localitate.getCotaMinimaDeRisc() < localitate.getRau().getCotaMedie() && localitate.getCotaMaximaAdmisa() > localitate.getRau().getCotaMedie()).collect(Collectors.toList()));
        majorList.setAll(auxLocalitati.stream().filter((localitate) -> localitate.getCotaMaximaAdmisa() < localitate.getRau().getCotaMedie()).collect(Collectors.toList()));

        redusTable.setItems(redusList);
        mediuTable.setItems(mediuList);
        majorTable.setItems(majorList);
    }

    @Override
    public void update() {
        updateAvertizari();
    }

    public void calculeazaRisc() {
        String numeRau = rauTextField.getText();
        String cmdr = cmdrTextField.getText();
        String cma = cmaTextField.getText();

        Rau rau = service.getRau(numeRau);
        if (rau == null) {
            Alert alert = new Alert(Alert.AlertType.ERROR, "Acest rau nu exista", ButtonType.OK);
            alert.show();
            return;
        }
        try {
            Integer cotaMinima = Integer.parseInt(cmdr);
            Integer cotaMaxima = Integer.parseInt(cma);

            if (rau.getCotaMedie() < cotaMinima) {
                riscLabel.setText("Risc: redus");
            } else if (rau.getCotaMedie() < cotaMaxima) {
                riscLabel.setText("Risc: medium");
            } else {
                riscLabel.setText("Risc: major");
            }
        } catch (Exception e) {
            Alert alert = new Alert(Alert.AlertType.ERROR, "Cota trebuie sa fie numar", ButtonType.OK);
            alert.show();
        }
    }
}
