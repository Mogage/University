package clinica.controllers;

import clinica.domain.DTOMedicSectie;
import clinica.domain.Medic;
import clinica.domain.Sectie;
import clinica.service.Service;
import javafx.collections.FXCollections;
import javafx.collections.ObservableList;
import javafx.fxml.FXML;
import javafx.scene.control.TableColumn;
import javafx.scene.control.TableView;
import javafx.scene.control.cell.PropertyValueFactory;

import java.util.ArrayList;
import java.util.List;

public class MainController {

    @FXML
    TableView<DTOMedicSectie> sectieTableView;

    @FXML
    TableColumn<DTOMedicSectie, String> numeSectie;

    @FXML
    TableColumn<DTOMedicSectie, Integer> pretSectie;

    @FXML
    TableColumn<DTOMedicSectie, Long> numeSefDeSectie;

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
            if(null != medic)
                medicSectieList.add(new DTOMedicSectie(sectie.getNume(), medic.getNume(), sectie.getPretPerConsultatie()));
        }

        sectiiList.setAll(medicSectieList);
        sectieTableView.setItems(sectiiList);
    }
}