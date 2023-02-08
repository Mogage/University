package ati.controllers;

import ati.domain.Pat;
import ati.service.Service;
import ati.utils.Observer;
import javafx.fxml.FXML;
import javafx.scene.control.*;

import java.security.spec.ECField;
import java.util.HashMap;
import java.util.Map;

public class PaturiController implements Observer {
    @FXML
    public TextField cnpTextField;
    @FXML
    public Button elibereazaPatButton;
    @FXML
    public Label paturiOcupateLabel;
    @FXML
    public Label ticLabel;
    @FXML
    public Label timLabel;
    @FXML
    public Label tiipLabel;

    private Service service;

    public void initialise(Service service) {
        this.service = service;
        updatePaturi();
    }

    private void updatePaturi() {
        Iterable<Pat> paturi = service.getPaturi();
        Map<String, Integer> numarPaturiLibere = new HashMap<>();

        numarPaturiLibere.put("TIC", 0);
        numarPaturiLibere.put("TIM", 0);
        numarPaturiLibere.put("TIIP", 0);

        for (Pat pat : paturi) {
            if (pat.getCnpPacient() == null) {
                numarPaturiLibere.put(pat.getTip(), numarPaturiLibere.get(pat.getTip()) + 1);
            }
        }

        int paturiOcupate = 30 - numarPaturiLibere.get("TIC") - numarPaturiLibere.get("TIM") - numarPaturiLibere.get("TIIP");
        paturiOcupateLabel.setText("Paturi ocupate: " + paturiOcupate);
        ticLabel.setText("TIC " + numarPaturiLibere.get("TIC").toString() + " paturi libere");
        timLabel.setText("TIM " + numarPaturiLibere.get("TIM").toString() + " paturi libere");
        tiipLabel.setText("TIIP " + numarPaturiLibere.get("TIIP").toString() + " paturi libere");
    }

    @Override
    public void update() {
        updatePaturi();
    }

    public void elibereazaPat() {
        String cnp = cnpTextField.getText();
        try {
            service.elibereaza(cnp);
        } catch (Exception e) {
            Alert alert = new Alert(Alert.AlertType.ERROR, e.getMessage(), ButtonType.OK);
            alert.show();
        }
    }
}