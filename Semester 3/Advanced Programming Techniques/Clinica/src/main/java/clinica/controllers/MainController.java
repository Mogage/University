package clinica.controllers;

import clinica.Main;
import clinica.domain.DTOMedicSectie;
import clinica.domain.Medic;
import clinica.domain.Sectie;
import clinica.service.Service;
import clinica.utils.Observable;
import clinica.utils.Observer;
import javafx.collections.FXCollections;
import javafx.collections.ObservableList;
import javafx.fxml.FXML;
import javafx.fxml.FXMLLoader;
import javafx.scene.Scene;
import javafx.scene.control.*;
import javafx.scene.control.cell.PropertyValueFactory;
import javafx.stage.Stage;

import java.io.IOException;
import java.util.*;

public class MainController {

    @FXML
    TableView<DTOMedicSectie> sectieTableView;

    @FXML
    TableColumn<DTOMedicSectie, String> numeSectie;

    @FXML
    TableColumn<DTOMedicSectie, Integer> pretSectie;

    @FXML
    TableColumn<DTOMedicSectie, Long> numeSefDeSectie;

    @FXML
    Button programareButton;

    private Service service;

    private final ObservableList<DTOMedicSectie> sectiiList = FXCollections.observableArrayList();

    public void initialise(Service service) {
        this.service = service;

        initSectii();
    }

    private void initSectii() {
        numeSectie.setCellValueFactory(new PropertyValueFactory<>("numeSectie"));
        numeSefDeSectie.setCellValueFactory(new PropertyValueFactory<>("numeSefDeSectie"));
        pretSectie.setCellValueFactory(new PropertyValueFactory<>("pret"));
        updateSectii();
    }

    private void updateSectii() {
        Iterable<Sectie> sectii = service.getSectii();
        List<DTOMedicSectie> medicSectieList = new ArrayList<>();

        for (Sectie sectie : sectii) {
            Medic medic = service.findMedic(sectie.getIdSefDeSectie());
            if(null != medic) {
                medicSectieList.add(new DTOMedicSectie(sectie.getId(), sectie.getNume(), medic.getNume(), sectie.getPretPerConsultatie()));
            }
        }

        sectiiList.setAll(medicSectieList);
        sectieTableView.setItems(sectiiList);
    }

    public void openProgramare() {
        DTOMedicSectie medicSectie = sectieTableView.getSelectionModel().getSelectedItem();
        if (null == medicSectie) {
            Alert alert = new Alert(Alert.AlertType.ERROR, "Alege o sectie", ButtonType.OK);
            alert.show();
            return;
        }
        Sectie sectie = service.findSectie(medicSectie.getId());
        FXMLLoader fxmlLoader = new FXMLLoader(Main.class.getResource("ProgramareView.fxml"));
        try {
            Scene scene = new Scene(fxmlLoader.load(), 450, 405);
            ProgramareController programareController = fxmlLoader.getController();
            programareController.initialise(service, sectie);
            Stage stage = new Stage();
            stage.setTitle(sectie.getNume());
            stage.setScene(scene);
            stage.setResizable(false);
            stage.show();
        } catch (Exception e) {
            Alert alert = new Alert(Alert.AlertType.ERROR, e.getMessage(), ButtonType.OK);
            alert.show();
        }
    }

}