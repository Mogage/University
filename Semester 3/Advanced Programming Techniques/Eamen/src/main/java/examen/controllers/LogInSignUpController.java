package examen.controllers;

import javafx.collections.ObservableList;
import javafx.fxml.FXML;

import examen.domain.Persoana;
import examen.service.Service;
import examen.utils.Constants;
import javafx.collections.FXCollections;
import javafx.fxml.FXMLLoader;
import javafx.scene.Scene;
import javafx.scene.control.*;
import javafx.scene.control.cell.PropertyValueFactory;
import javafx.stage.Stage;

import java.util.ArrayList;
import java.util.List;

public class LogInSignUpController {
    @FXML
    public TextField numeTextField;
    @FXML
    public TextField prenumeTextField;
    @FXML
    public TextField usernameTextField;
    @FXML
    public PasswordField parolaTextField;
    @FXML
    public TextField stradaTextField;
    @FXML
    public TextField numarStradaTextField;
    @FXML
    public TextField telefonTextField;
    @FXML
    public ComboBox<Constants.ORASE> orasComboBox;
    @FXML
    public Button inregistrareButton;
    @FXML
    public TableView<Persoana> utilizatoriTable;
    @FXML
    public TableColumn<Persoana, String> usernameColumn;

    private Service service;

    private final ObservableList<Persoana> persoanaObservableList = FXCollections.observableArrayList();

    public void initialise(Service service) {
        this.service = service;
        orasComboBox.setItems(FXCollections.observableArrayList(Constants.ORASE.values()));
        utilizatoriTable.getSelectionModel().selectedItemProperty().addListener(event -> conectare());
        initUsernames();
    }

    private void initUsernames() {
        usernameColumn.setCellValueFactory(new PropertyValueFactory<>("username"));
        updateUsernames();
    }

    private void updateUsernames() {
        Iterable<Persoana> persoane = service.getPersoane();
        List<Persoana> auxPersoane = new ArrayList<>();

        for (Persoana persoana : persoane) {
            auxPersoane.add(persoana);
        }

        persoanaObservableList.setAll(auxPersoane);
        utilizatoriTable.setItems(persoanaObservableList);
    }

    public void inregistrare() {
        String nume = numeTextField.getText();
        String prenume = prenumeTextField.getText();
        String username = usernameTextField.getText();
        String parola = parolaTextField.getText();
        String oras = (orasComboBox.getValue() == null) ? "" : orasComboBox.getValue().name();
        String strada = stradaTextField.getText();
        String numarStrada = numarStradaTextField.getText();
        String telefon = telefonTextField.getText();
        try {
            service.addPerson(nume, prenume, username, parola, oras, strada, numarStrada, telefon);
        } catch (Exception e) {
            Alert alert = new Alert(Alert.AlertType.ERROR, e.getMessage(), ButtonType.OK);
            alert.show();
        }
        updateUsernames();
        resetForm();
    }

    private void resetForm() {
        numeTextField.setText("");
        prenumeTextField.setText("");
        usernameTextField.setText("");
        parolaTextField.setText("");
        orasComboBox.valueProperty().set(null);
        stradaTextField.setText("");
        numarStradaTextField.setText("");
        telefonTextField.setText("");
    }

    public void conectare() {
        Persoana persoana = utilizatoriTable.getSelectionModel().getSelectedItem();
        if (persoana == null) {
            return;
        }

        Scene scene;
        FXMLLoader loader = new FXMLLoader(examen.Main.class.getResource("ajutaView.fxml"));
        try {
            scene = new Scene(loader.load(), 800, 450);
        } catch (Exception e) {
            Alert alert = new Alert(Alert.AlertType.ERROR, e.getMessage(), ButtonType.OK);
            alert.show();
            return;
        }

        AjutaController ajutaController = loader.getController();
        ajutaController.initialise(service, persoana);
        service.addObserver(ajutaController);

        Stage newStage = new Stage();
        newStage.setScene(scene);
        newStage.setTitle("Fapte bune");
        newStage.show();

    }
}