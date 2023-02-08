package anar.service;

import anar.domain.Localitate;
import anar.domain.Rau;
import anar.repository.LocalitatiRepository;
import anar.repository.RauriRepository;
import anar.utils.Observable;

import java.util.Objects;

public class Service implements Observable {
    RauriRepository rauriRepository;
    LocalitatiRepository localitatiRepository;

    public Service(RauriRepository rauriRepository, LocalitatiRepository localitatiRepository) {
        this.rauriRepository = rauriRepository;
        this.localitatiRepository = localitatiRepository;
    }

    public Iterable<Rau> getRauri() {
        return rauriRepository.getAll();
    }

    public Iterable<Localitate> getLocalitati() {
        return localitatiRepository.getAll();
    }

    public void updateRau(Rau rau, Integer cota) {
        rauriRepository.update(rau, cota);
        Iterable<Localitate> localitati = localitatiRepository.getAll();

        for (Localitate localitate : localitati) {
            if (Objects.equals(localitate.getRau().getNume(), rau.getNume())) {
                localitatiRepository.update(localitate, cota);
            }
        }
        notifyObservers();
    }

    public Rau getRau(String numeRau) {
        return rauriRepository.findAfterName(numeRau);
    }
}
