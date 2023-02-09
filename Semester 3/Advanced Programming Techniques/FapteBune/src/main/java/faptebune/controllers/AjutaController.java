package faptebune.controllers;

import faptebune.domain.Nevoie;
import faptebune.domain.Persoana;
import faptebune.service.Service;
import faptebune.utils.observer.Observer;
import javafx.collections.FXCollections;
import javafx.collections.ObservableList;
import javafx.fxml.FXML;
import javafx.scene.control.*;
import javafx.scene.control.cell.PropertyValueFactory;

import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

public class AjutaController implements Observer {
    @FXML
    public Label numeUtilizatorLabel;
    @FXML
    public TableView<Nevoie> nevoiTable;
    @FXML
    public TableView<Nevoie> nevoieRezolvateTable;
    @FXML
    public TableColumn<Nevoie, String> titluNevoieColumn;
    @FXML
    public TableColumn<Nevoie, String> descriereNevoieColumn;
    @FXML
    public TableColumn<Nevoie, LocalDateTime> deadlineNevoieColumn;
    @FXML
    public TableColumn<Nevoie, String> statusNevoieColumn;
    @FXML
    public TableColumn<Nevoie, Long> omInNevoieNevoieColumn;
    @FXML
    public TableColumn<Nevoie, Long> omSalvatorNevoieColumn;
    @FXML
    public TableColumn<Nevoie, String> titluRezolvatColumn;
    @FXML
    public TableColumn<Nevoie, LocalDateTime> deadlineRezolvatColumn;
    @FXML
    public TableColumn<Nevoie, Long> omInNevoieRezolvatColumn;
    @FXML
    public Button rezolvaButton;
    @FXML
    public TextField titluTextField;
    @FXML
    public TextField descriereTextField;
    @FXML
    public DatePicker data;
    @FXML
    public Spinner<Integer> ora;
    @FXML
    public Spinner<Integer> minut;
    @FXML
    public Button submitButton;

    private Service service;
    private Persoana persoanaConectata;

    private final ObservableList<Nevoie> nevoieObservableList = FXCollections.observableArrayList();
    private final ObservableList<Nevoie> nevoieRezolvatObservableList = FXCollections.observableArrayList();

    public void initialise(Service service, Persoana persoanaConectata) {
        this.service = service;
        this.persoanaConectata = persoanaConectata;
        numeUtilizatorLabel.setText(persoanaConectata.getUsername());

        initNevoi();
        initNevoieRezolvate();
    }

    private void initNevoi() {
        titluNevoieColumn.setCellValueFactory(new PropertyValueFactory<>("titlu"));
        descriereNevoieColumn.setCellValueFactory(new PropertyValueFactory<>("descriere"));
        deadlineNevoieColumn.setCellValueFactory(new PropertyValueFactory<>("deadline"));
        statusNevoieColumn.setCellValueFactory(new PropertyValueFactory<>("status"));
        omInNevoieNevoieColumn.setCellValueFactory(new PropertyValueFactory<>("omInNevoie"));
        omSalvatorNevoieColumn.setCellValueFactory(new PropertyValueFactory<>("omSalvator"));
        updateNevoi();
    }

    private void updateNevoi() {
        Iterable<Nevoie> nevoi = service.getNevoi();
        List<Nevoie> auxNevoi = new ArrayList<>();

        for (Nevoie nevoie : nevoi) {
            if (!Objects.equals(nevoie.getOmInNevoie(), persoanaConectata.getId()) &&
                    nevoie.getOmSalvator() == 0 &&
                    Objects.equals(service.getOras(nevoie.getOmInNevoie()), persoanaConectata.getOras())
            ) {
                auxNevoi.add(nevoie);
            }
        }

        nevoieObservableList.setAll(auxNevoi);
        nevoiTable.setItems(nevoieObservableList);
    }

    private void initNevoieRezolvate() {
        titluRezolvatColumn.setCellValueFactory(new PropertyValueFactory<>("titlu"));
        deadlineRezolvatColumn.setCellValueFactory(new PropertyValueFactory<>("deadline"));
        omInNevoieRezolvatColumn.setCellValueFactory(new PropertyValueFactory<>("omInNevoie"));
        updateNevoiRezolvate();
    }

    private void updateNevoiRezolvate() {
        Iterable<Nevoie> nevoi = service.getNevoi();
        List<Nevoie> auxNevoi = new ArrayList<>();

        for (Nevoie nevoie : nevoi) {
            if (Objects.equals(nevoie.getOmSalvator(), persoanaConectata.getId())) {
                auxNevoi.add(nevoie);
            }
        }

        nevoieRezolvatObservableList.setAll(auxNevoi);
        nevoieRezolvateTable.setItems(nevoieRezolvatObservableList);
    }

    public void submitForm() {
        String titlu = titluTextField.getText();
        String descriere = descriereTextField.getText();
        String dataNevoie = data.getValue().toString();
        Integer oraNevoie = ora.getValue();
        Integer minutNevoie = minut.getValue();

        try {
            service.addNevoie(titlu, descriere, dataNevoie, oraNevoie, minutNevoie, persoanaConectata.getId());
        } catch (Exception e) {
            Alert alert = new Alert(Alert.AlertType.ERROR, e.getMessage(), ButtonType.OK);
            alert.show();
        }
        Alert alert = new Alert(Alert.AlertType.CONFIRMATION, "Nevoie trimisa cu succes", ButtonType.OK);
        alert.show();
    }

    public void rezolvaNevoie() {
        Nevoie nevoie = nevoiTable.getSelectionModel().getSelectedItem();
        if (nevoie == null) {
            Alert alert = new Alert(Alert.AlertType.ERROR, "Nu a fost aleasa nicio nevoie.", ButtonType.OK);
            alert.show();
            return;
        }
        service.updateNevoie(nevoie, persoanaConectata.getId(), "Erou gasit!");
    }

    @Override
    public void update() {
        updateNevoi();
        updateNevoiRezolvate();
    }
}
