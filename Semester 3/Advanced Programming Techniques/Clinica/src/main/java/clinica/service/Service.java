package clinica.service;

import clinica.domain.Consultatie;
import clinica.domain.Medic;
import clinica.domain.Sectie;

public interface Service{

    Iterable<Sectie> getSectii();
    
    Iterable<Consultatie> getConsultatii();

    Iterable<Medic> getMedici();

    Iterable<Consultatie> getConsultatiiSectie(Sectie sectie);

    Sectie findSectie(Long id);

    Medic findMedic(Long id);

    void programare(Consultatie consultatie);

    void sterge(Consultatie consultatie);
}
