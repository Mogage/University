package examen.controllers;

import examen.domain.DTONevoiePersoana;
import examen.domain.Nevoie;
import examen.domain.Persoana;
import examen.service.Service;
import examen.utils.observer.Observer;
import javafx.collections.FXCollections;
import javafx.collections.ObservableList;
import javafx.fxml.FXML;
import javafx.scene.control.*;
import javafx.scene.control.cell.PropertyValueFactory;

import java.time.LocalDate;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

public class AjutaController implements Observer {
    @FXML
    public Label numeUtilizatorLabel;
    @FXML
    public TableView<DTONevoiePersoana> nevoiTable;
    @FXML
    public TableView<DTONevoiePersoana> nevoieRezolvateTable;
    @FXML
    public TableColumn<DTONevoiePersoana, String> titluNevoieColumn;
    @FXML
    public TableColumn<DTONevoiePersoana, String> descriereNevoieColumn;
    @FXML
    public TableColumn<DTONevoiePersoana, String> deadlineNevoieColumn;
    @FXML
    public TableColumn<DTONevoiePersoana, String> statusNevoieColumn;
    @FXML
    public TableColumn<DTONevoiePersoana, String> omInNevoieNevoieColumn;
    @FXML
    public TableColumn<DTONevoiePersoana, String> titluRezolvatColumn;
    @FXML
    public TableColumn<DTONevoiePersoana, String> deadlineRezolvatColumn;
    @FXML
    public TableColumn<DTONevoiePersoana, String> omInNevoieRezolvatColumn;
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
    @FXML
    public Label orasLabel;

    private Service service;
    private Persoana persoanaConectata;

    private final ObservableList<DTONevoiePersoana> nevoieObservableList = FXCollections.observableArrayList();
    private final ObservableList<DTONevoiePersoana> nevoieRezolvatObservableList = FXCollections.observableArrayList();

    public void initialise(Service service, Persoana persoanaConectata) {
        this.service = service;
        this.persoanaConectata = persoanaConectata;
        numeUtilizatorLabel.setText(persoanaConectata.getUsername());
        orasLabel.setText("Oameni de ajutat din " + persoanaConectata.getOras());
        data.setValue(LocalDate.now());

        initNevoi();
        initNevoieRezolvate();
    }

    private void initNevoi() {
        titluNevoieColumn.setCellValueFactory(new PropertyValueFactory<>("titlu"));
        descriereNevoieColumn.setCellValueFactory(new PropertyValueFactory<>("descriere"));
        deadlineNevoieColumn.setCellValueFactory(new PropertyValueFactory<>("deadline"));
        statusNevoieColumn.setCellValueFactory(new PropertyValueFactory<>("status"));
        omInNevoieNevoieColumn.setCellValueFactory(new PropertyValueFactory<>("omInNevoie"));
        updateNevoi();
    }

    private void updateNevoi() {
        Iterable<Nevoie> nevoi = service.getNevoi();
        List<DTONevoiePersoana> auxNevoi = new ArrayList<>();

        for (Nevoie nevoie : nevoi) {
            Persoana omInNevoie = service.getPersoana(nevoie.getOmInNevoie());
            if (!Objects.equals(nevoie.getOmInNevoie(), persoanaConectata.getId()) &&
                    nevoie.getOmSalvator() == 0 &&
                    Objects.equals(omInNevoie.getOras(), persoanaConectata.getOras())
            ) {
                DTONevoiePersoana dtoNevoiePersoana = new DTONevoiePersoana(nevoie.getTitlu(), nevoie.getDescriere(),
                        nevoie.getDeadline(), omInNevoie.getNume(), nevoie.getStatus());
                dtoNevoiePersoana.setId(nevoie.getId());
                auxNevoi.add(dtoNevoiePersoana);
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
        List<DTONevoiePersoana> auxNevoi = new ArrayList<>();

        for (Nevoie nevoie : nevoi) {
            if (Objects.equals(nevoie.getOmSalvator(), persoanaConectata.getId())) {
                Persoana omInNevoie = service.getPersoana(nevoie.getOmInNevoie());
                DTONevoiePersoana dtoNevoiePersoana = new DTONevoiePersoana(nevoie.getTitlu(), nevoie.getDescriere(),
                        nevoie.getDeadline(), omInNevoie.getNume(), nevoie.getStatus());
                dtoNevoiePersoana.setId(nevoie.getId());
                auxNevoi.add(dtoNevoiePersoana);
            }
        }

        nevoieRezolvatObservableList.setAll(auxNevoi);
        nevoieRezolvateTable.setItems(nevoieRezolvatObservableList);
    }

    public void submitForm() {
        String titlu = titluTextField.getText();
        String descriere = descriereTextField.getText();
        String dataNevoie = (data.getValue() == null) ? "" : data.getValue().toString();
        Integer oraNevoie = ora.getValue();
        Integer minutNevoie = minut.getValue();

        try {
            service.addNevoie(titlu, descriere, dataNevoie, oraNevoie, minutNevoie, persoanaConectata.getId());
            resetForm();
        } catch (Exception e) {
            Alert alert = new Alert(Alert.AlertType.ERROR, e.getMessage(), ButtonType.OK);
            alert.show();
            return;
        }
        Alert alert = new Alert(Alert.AlertType.CONFIRMATION, "Nevoie trimisa cu succes", ButtonType.OK);
        alert.show();
    }

    private void resetForm() {
        titluTextField.setText("");
        descriereTextField.setText("");
        data.setValue(LocalDate.now());
        ora.getValueFactory().setValue(0);
        minut.getValueFactory().setValue(0);
    }

    public void rezolvaNevoie() {
        DTONevoiePersoana nevoie = nevoiTable.getSelectionModel().getSelectedItem();
        if (nevoie == null) {
            Alert alert = new Alert(Alert.AlertType.ERROR, "Nu a fost aleasa nicio nevoie.", ButtonType.OK);
            alert.show();
            return;
        }
        service.updateNevoie(nevoie.getId(), persoanaConectata.getId(), "Erou gasit!");
    }

    @Override
    public void update() {
        updateNevoi();
        updateNevoiRezolvate();
    }
}
