package clinica.service;

import clinica.domain.Medic;
import clinica.domain.Sectie;
import clinica.repository.MediciRepository;
import clinica.repository.Repository;
import clinica.repository.SectiiRepository;

public class MainService implements Service {
    private final SectiiRepository sectiiRepository;
    private final MediciRepository mediciRepository;

    public MainService(Repository<Long, Sectie> sectiiRepository, Repository<Long, Medic> mediciRepository) {
        this.sectiiRepository = (SectiiRepository) sectiiRepository;
        this.mediciRepository = (MediciRepository) mediciRepository;
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
    public Sectie findSectie(Long id) {
        return this.sectiiRepository.findAfterId(id);
    }

    @Override
    public Medic findMedic(Long id) {
        return this.mediciRepository.findAfterId(id);
    }
}
