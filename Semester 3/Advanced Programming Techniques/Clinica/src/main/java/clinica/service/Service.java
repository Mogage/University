package clinica.service;

import clinica.domain.Medic;
import clinica.domain.Sectie;

public interface Service {

    Iterable<Sectie> getSectii();

    Iterable<Medic> getMedici();

    Sectie findSectie(Long id);

    Medic findMedic(Long id);
}
