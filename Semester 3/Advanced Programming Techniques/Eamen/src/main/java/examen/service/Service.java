package examen.service;

import examen.domain.Entity;
import examen.domain.Nevoie;
import examen.domain.Persoana;
import examen.domain.validators.Validator;
import examen.repository.NevoiRepository;
import examen.repository.PersoaneRepository;
import examen.utils.Constants;
import examen.utils.observer.Observable;

import java.time.LocalDateTime;
import java.util.Set;
import java.util.TreeSet;

public class Service implements Observable {
    Validator<Persoana> persoanaValidator;
    Validator<Nevoie> nevoieValidator;
    PersoaneRepository persoaneRepository;
    NevoiRepository nevoiRepository;

    public Service(Validator<Persoana> persoanaValidator, Validator<Nevoie> nevoieValidator,
                   PersoaneRepository persoaneRepository, NevoiRepository nevoiRepository) {
        this.persoanaValidator = persoanaValidator;
        this.nevoieValidator = nevoieValidator;
        this.persoaneRepository = persoaneRepository;
        this.nevoiRepository = nevoiRepository;
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

    public void addPerson(String nume, String prenume, String username, String parola, String oras, String strada,
                          String numarStrada, String telefon) throws Exception {
        Persoana persoana = new Persoana(nume, prenume, username, parola, oras, strada, numarStrada, telefon);
        persoana.setId(getId(persoaneRepository.getAll()));
        this.persoanaValidator.validate(persoana);
        this.persoaneRepository.save(persoana);
    }

    public void addNevoie(String titlu, String descriere, String data, Integer Ora, Integer Minut, Long omInNevoie) throws Exception {
        String oraToParse;
        String minutToParse;
        if (Ora < 10) {
            oraToParse = "0" + Ora;
        } else oraToParse = Ora.toString();
        if (Minut < 10) {
            minutToParse = "0" + Minut;
        } else minutToParse = Minut.toString();
        String deadlineToParse = data + " " + oraToParse + ":" + minutToParse;
        LocalDateTime deadline = LocalDateTime.parse(deadlineToParse, Constants.DATE_TIME_FORMATTER);
        Nevoie nevoie = new Nevoie(titlu, descriere, deadline, omInNevoie, null, "Caut erou!");
        nevoie.setId(getId(nevoiRepository.getAll()));
        this.nevoieValidator.validate(nevoie);
        this.nevoiRepository.save(nevoie);
        notifyObservers();
    }

    public void updateNevoie(Long nevoieId, Long omSalvator, String status) {
        Nevoie nevoie = nevoiRepository.findAfterId(nevoieId);
        nevoie.setOmSalvator(omSalvator);
        nevoie.setStatus(status);
        nevoiRepository.update(nevoie);
        notifyObservers();
    }

    public Iterable<Persoana> getPersoane() {
        return persoaneRepository.getAll();
    }

    public Iterable<Nevoie> getNevoi() {
        return nevoiRepository.getAll();
    }

    public Persoana getPersoana(Long id) {
        return persoaneRepository.findAfterId(id);
    }
}
