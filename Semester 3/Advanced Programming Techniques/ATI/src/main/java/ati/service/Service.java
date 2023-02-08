package ati.service;

import ati.domain.Pacient;
import ati.domain.Pat;
import ati.repository.PacientiRepository;
import ati.repository.PaturiRepository;
import ati.utils.Observable;

import java.util.ArrayList;
import java.util.List;

public class Service implements Observable {

    private final PaturiRepository paturiRepository;
    private final PacientiRepository pacientiRepository;

    public Service(PaturiRepository paturiRepository, PacientiRepository pacientiRepository) {
        this.paturiRepository = paturiRepository;
        this.pacientiRepository = pacientiRepository;
    }

    public Iterable<Pat> getPaturi() {
        return paturiRepository.getAll();
    }

    public void ocupaPat(String tipPat, String cnp) throws Exception{
        Pat patLiber = paturiRepository.getPatLiber(tipPat);
        if (patLiber == null) {
            throw new Exception("Nu mai sunt paturi libere");
        }
        patLiber.setCnpPacient(cnp);
        paturiRepository.updatePat(patLiber);
        notifyObservers();
    }

    public Iterable<Pacient> getPacientiAsteptare() {
        Iterable<Pacient> pacienti = pacientiRepository.getAll();
        List<Pacient> pacientiAsteptare = new ArrayList<>();

        for (Pacient pacient : pacienti) {
            if (!paturiRepository.estePacientInternat(pacient.getCnp())) {
                pacientiAsteptare.add(pacient);
            }
        }
        return pacientiAsteptare;
    }

    public void elibereaza(String cnp) throws Exception {
        if (!paturiRepository.estePacientInternat(cnp)){
            throw new Exception("Acest pacient nu este internat");
        }
        Pat pat = paturiRepository.getPacientPat(cnp);
        pat.setCnpPacient(null);
        paturiRepository.updatePat(pat);
        pacientiRepository.sterge(cnp);
        notifyObservers();
    }
}
