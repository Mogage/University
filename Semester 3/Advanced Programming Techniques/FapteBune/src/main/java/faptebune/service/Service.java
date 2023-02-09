package faptebune.service;

import faptebune.domain.Entity;
import faptebune.domain.Nevoie;
import faptebune.domain.Persoana;
import faptebune.repository.NevoiRepository;
import faptebune.repository.PersoaneRepository;
import faptebune.utils.Constants;
import faptebune.utils.observer.Observable;

import java.time.LocalDateTime;
import java.util.Set;
import java.util.TreeSet;

public class Service implements Observable {
    PersoaneRepository persoaneRepository;
    NevoiRepository nevoiRepository;

    public Service(PersoaneRepository persoaneRepository, NevoiRepository nevoiRepository) {
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
        this.persoaneRepository.save(persoana);
    }

    public void addNevoie(String titlu, String descriere, String data, Integer Ora, Integer Minut, Long omInNevoie) throws Exception {
        String oraToParse;
        String minutToParse;
        if (Ora < 10) {
            oraToParse = "0" + Ora;
        }
        else oraToParse = Ora.toString();
        if(Minut < 10) {
            minutToParse = "0" + Minut;
        }
        else minutToParse = Minut.toString();
        String deadlineToParse = data + " " + oraToParse + ":" + minutToParse;
        LocalDateTime deadline = LocalDateTime.parse(deadlineToParse, Constants.DATE_TIME_FORMATTER);
        Nevoie nevoie = new Nevoie(titlu, descriere, deadline, omInNevoie, null, "Caut erou!");
        nevoie.setId(getId(nevoiRepository.getAll()));
        this.nevoiRepository.save(nevoie);
        notifyObservers();
    }

    public void updateNevoie(Nevoie nevoie, Long omSalvator, String status) {
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

    public String getOras(Long omInNevoie) {
        Persoana persoana = persoaneRepository.findAfterId(omInNevoie);
        return persoana.getOras();
    }
}
