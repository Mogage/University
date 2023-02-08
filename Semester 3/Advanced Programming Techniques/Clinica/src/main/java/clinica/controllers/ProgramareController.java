package clinica.controllers;

import clinica.domain.Consultatie;
import clinica.domain.Medic;
import clinica.domain.Sectie;
import clinica.service.Service;
import clinica.utils.Observable;
import javafx.collections.FXCollections;
import javafx.collections.ObservableList;
import javafx.fxml.FXML;
import javafx.scene.control.*;
import javafx.scene.control.cell.PropertyValueFactory;

import java.time.LocalDate;
import java.time.LocalTime;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

public class ProgramareController {
    @FXML
    Label numeSectie;
    @FXML
    TableView<Medic> mediciTableView;
    @FXML
    TableColumn<Medic, String> numeMedic;
    @FXML
    Button programeazaButton;
    @FXML
    DatePicker data;
    @FXML
    TextField numePacient;
    @FXML
    TextField cnpPacient;
    @FXML
    Slider oraSlider;
    @FXML
    Label minuteLabel;
    @FXML
    Label oraLabel;
    @FXML
    Slider minuteSlider;

    private Service service;
    private Sectie sectie;

    private final ObservableList<Medic> mediciList = FXCollections.observableArrayList();

    public void initialise(Service service, Sectie sectie) {
        this.service = service;
        this.sectie = sectie;

        numeSectie.setText(sectie.getNume());
        oraSlider.valueProperty().addListener((observable, oldValue, newValue) -> oraLabel.textProperty().setValue(String.valueOf(newValue.intValue())));
        minuteSlider.valueProperty().addListener((observable, oldValue, newValue) -> minuteLabel.textProperty().setValue(String.valueOf(newValue.intValue())));
        initMedici();
    }

    private void initMedici() {
        numeMedic.setCellValueFactory(new PropertyValueFactory<>("nume"));
        updateMedici();
    }

    private void updateMedici() {
        Iterable<Medic> medici = service.getMedici();
        List<Medic> medicList = new ArrayList<>();

        for (Medic medic : medici) {
            if (Objects.equals(medic.getIdSectie(), sectie.getId())) {
                medicList.add(medic);
            }
        }

        mediciList.setAll(medicList);
        mediciTableView.setItems(mediciList);
    }

    private String checkErrors(String nume, String cnp, Medic medic) {
        String errors = "";
        if (nume.isEmpty()) {
            errors += "Name can't be empty.\n";
        }
        System.out.println(cnp.length());
        if (cnp.length() != 13) {
            errors += "CNP is not valid.\n";
        }
        if (medic == null) {
            errors += "You need to select a medic.\n";
        }
        return errors;
    }

    public void doProgramare() {
        String nume = numePacient.getText();
        String cnp = cnpPacient.getText();
        LocalDate date = data.getValue();
        Medic medic = mediciTableView.getSelectionModel().getSelectedItem();
        String errors = checkErrors(nume, cnp, medic);
        if (!errors.isEmpty()) {
            Alert alert = new Alert(Alert.AlertType.ERROR, errors, ButtonType.OK);
            alert.show();
            return;
        }
        String ora;
        if (oraLabel.getText().length() == 1)
            ora = "0" + oraLabel.getText() + ":";
        else
            ora = oraLabel.getText() + ":";
        if (minuteLabel.getText().length() == 1)
            ora = ora + "0" + minuteLabel.getText();
        else
            ora = ora + minuteLabel.getText();
        LocalTime time = LocalTime.parse(ora, DateTimeFormatter.ofPattern("HH:mm"));
        Long idMedic = medic.getId();
        Consultatie consultatie = new Consultatie(idMedic, nume, cnp, date, time);
        service.programare(consultatie);
        Alert alert = new Alert(Alert.AlertType.INFORMATION, "Programare cu succes", ButtonType.OK);
        alert.show();
    }
}
