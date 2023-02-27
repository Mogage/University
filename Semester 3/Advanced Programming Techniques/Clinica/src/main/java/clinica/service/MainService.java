package clinica.service;

import clinica.domain.Consultatie;
import clinica.domain.Entity;
import clinica.domain.Medic;
import clinica.domain.Sectie;
import clinica.repository.ConsultatiiRepository;
import clinica.repository.MediciRepository;
import clinica.repository.Repository;
import clinica.repository.SectiiRepository;
import clinica.utils.Observable;

import java.util.*;

public class MainService implements Service, Observable {
    private final SectiiRepository sectiiRepository;
    private final MediciRepository mediciRepository;

    private final ConsultatiiRepository consultatiiRepository;

    public MainService(Repository<Long, Sectie> sectiiRepository, Repository<Long, Medic> mediciRepository, Repository<Long, Consultatie> consultatiiRepository) {
        this.sectiiRepository = (SectiiRepository) sectiiRepository;
        this.mediciRepository = (MediciRepository) mediciRepository;
        this.consultatiiRepository = (ConsultatiiRepository) consultatiiRepository;
    }

    @Override
    public Iterable<Sectie> getSectii() {
        return this.sectiiRepository.getAll();
    }

    @Override
    public Iterable<Medic> getMedici() {
        return this.mediciRepository.getAll();
    }

    @Override
    public Iterable<Consultatie> getConsultatii() {
        return this.consultatiiRepository.getAll();
    }

    @Override
    public Iterable<Consultatie> getConsultatiiSectie(Sectie sectie) {
        Iterable<Medic> medici = mediciRepository.getAll();
        Iterable<Consultatie> consultatii = consultatiiRepository.getAll();
        List<Consultatie> rezultat = new ArrayList<>();
        for (Medic medic : medici) {
            if (Objects.equals(medic.getIdSectie(), sectie.getId())) {
                for (Consultatie consultatie : consultatii) {
                    if (Objects.equals(consultatie.getIdMedic(), medic.getId())) {
                        rezultat.add(consultatie);
                    }
                }
            }
        }
        return rezultat;
    }

    @Override
    public Sectie findSectie(Long id) {
        return this.sectiiRepository.findAfterId(id);
    }

    @Override
    public Medic findMedic(Long id) {
        return this.mediciRepository.findAfterId(id);
    }

    private Long getId(Iterable<? extends Entity<Long>> entities) {
        Set<Long> distinct = new TreeSet<>();
        long id = 1L;

        for (Entity<Long> entity : entities) {
            distinct.add(entity.getId());
        }

        while (true) {
            if (!distinct.contains(id)) {
                return id;
            }
            id = id + 1;
        }
    }

    @Override
    public void programare(Consultatie consultatie) {
        consultatie.setId(getId(consultatiiRepository.getAll()));
        this.consultatiiRepository.add(consultatie);
        notifyObservers();
    }

    @Override
    public void sterge(Consultatie consultatie) {
        this.consultatiiRepository.delete(consultatie.getId());
        notifyObservers();
    }
}
