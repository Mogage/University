package clinica.controllers;

import clinica.domain.Consultatie;
import clinica.domain.Medic;
import clinica.domain.Sectie;
import clinica.service.Service;
import clinica.utils.Observer;
import javafx.collections.FXCollections;
import javafx.collections.ObservableList;
import javafx.scene.control.*;
import javafx.scene.control.cell.PropertyValueFactory;

import java.time.LocalDate;
import java.time.LocalTime;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.stream.Collectors;

public class MedicController implements Observer {
    public Label numeMedic;
    public TableView<Consultatie> consultatieTableView;
    public TableColumn<Consultatie, String> numePacient;
    public TableColumn<Consultatie, String> cnpPacient;
    public TableColumn<Consultatie, LocalDate> data;
    public TableColumn<Consultatie, LocalTime> ora;
    public Button stergeProgramareButton;
    public TableView<Consultatie> consultatieSectie;
    public TableColumn<Consultatie, String> numePacientSectie;
    public TableColumn<Consultatie, String> cnpPacientSectie;
    public TableColumn<Consultatie, LocalDate> dataSectie;
    public TableColumn<Consultatie, LocalTime> oraSectie;
    public Label sectieLabel;
    public Label numeSefLabel;
    public Label actualNumeSefLabel;

    private Service service;
    private Medic medic;

    private final ObservableList<Consultatie> consultatiiList = FXCollections.observableArrayList();
    private final ObservableList<Consultatie> consultatiiListSectie = FXCollections.observableArrayList();

    public void initialise(Service service, Medic medic) {
        this.service = service;
        this.medic = medic;

        numeMedic.setText(medic.getNume());
        initConsultatii();
        initConsultatiiSectie();
    }

    private void initConsultatii() {
        numePacient.setCellValueFactory(new PropertyValueFactory<>("numePacient"));
        cnpPacient.setCellValueFactory(new PropertyValueFactory<>("cnpPacient"));
        data.setCellValueFactory(new PropertyValueFactory<>("data"));
        ora.setCellValueFactory(new PropertyValueFactory<>("ora"));
        updateConsultatii();
    }

    private void initConsultatiiSectie() {
        numePacientSectie.setCellValueFactory(new PropertyValueFactory<>("numePacient"));
        cnpPacientSectie.setCellValueFactory(new PropertyValueFactory<>("cnpPacient"));
        dataSectie.setCellValueFactory(new PropertyValueFactory<>("data"));
        oraSectie.setCellValueFactory(new PropertyValueFactory<>("ora"));
        updateConsultatiiSectie();
    }

    private boolean checkConsultatie(Consultatie consultatie, boolean isMedic) {
        if (isMedic && !Objects.equals(consultatie.getIdMedic(), medic.getId())) {
            return false;
        }
        if (consultatie.getData().isEqual(LocalDate.now()) && consultatie.getOra().isBefore(LocalTime.now())) {
            return false;
        }
        return consultatie.getData().isAfter(LocalDate.now());
    }

    private void updateConsultatiiSectie() {

        Iterable<Sectie> sectii = service.getSectii();

        boolean sefDeSectie = false;
        Sectie sectieDeAfisat = null;

        for (Sectie sectie : sectii) {
            if (Objects.equals(sectie.getIdSefDeSectie(), medic.getId())) {
                sefDeSectie = true;
                sectieDeAfisat = sectie;
                break;
            }
        }

        if (sefDeSectie) {
            numeSefLabel.setVisible(false);
            actualNumeSefLabel.setVisible(false);
            Iterable<Consultatie> consultatii = service.getConsultatiiSectie(sectieDeAfisat);
            List<Consultatie> consultatieList = new ArrayList<>();

            for (Consultatie consultatie : consultatii) {
                if (checkConsultatie(consultatie, false)) {
                    consultatieList.add(consultatie);
                }
            }

            setData(consultatieList, consultatiiListSectie, consultatieSectie);
        } else {
            consultatieSectie.setVisible(false);
            sectieLabel.setVisible(false);
            numeSefLabel.setVisible(true);
            actualNumeSefLabel.setVisible(true);
            actualNumeSefLabel.setText(service.findMedic(service.findSectie(medic.getIdSectie()).getIdSefDeSectie()).getNume());
        }
    }

    private void updateConsultatii() {
        Iterable<Consultatie> consultatii = service.getConsultatii();
        List<Consultatie> consultatieList = new ArrayList<>();

        for (Consultatie consultatie : consultatii) {
            if (checkConsultatie(consultatie, true)) {
                consultatieList.add(consultatie);
            }
        }

        setData(consultatieList, consultatiiList, consultatieTableView);
    }

    private void setData(List<Consultatie> consultatieList, ObservableList<Consultatie> consultatiiListSectie, TableView<Consultatie> consultatieSectie) {
        consultatiiListSectie.setAll(consultatieList.stream().sorted((entity1, entity2) -> {
            if (!entity1.getData().isEqual(entity2.getData())) {
                return entity1.getData().compareTo(entity2.getData());
            }
            return entity1.getOra().compareTo(entity2.getOra());
        }).collect(Collectors.toList()));
        consultatieSectie.setItems(consultatiiListSectie);
    }

    @Override
    public void update() {
        updateConsultatii();
        updateConsultatiiSectie();
    }

    public void deleteProgramare() {
        Consultatie consultatie = consultatieTableView.getSelectionModel().getSelectedItem();
        if (null == consultatie) {
            Alert alert = new Alert(Alert.AlertType.ERROR, "Nu a fost aleasa nicio consulatie de sters", ButtonType.OK);
            alert.show();
        }
        service.sterge(consultatie);
    }
}
