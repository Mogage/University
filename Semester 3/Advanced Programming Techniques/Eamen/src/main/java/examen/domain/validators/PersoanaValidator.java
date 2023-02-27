package examen.domain.validators;

import examen.domain.Persoana;

public class PersoanaValidator implements Validator<Persoana> {
    private static final PersoanaValidator instance = new PersoanaValidator();

    private PersoanaValidator() {}

    public static PersoanaValidator getInstance() {
        return instance;
    }
    
    @Override
    public void validate(Persoana entity) throws Exception {
        String err = "";

        if (entity.getNume().isBlank()) {
            err += "Numele nu poate fi gol.\n";
        }
        if (entity.getPrenume().isBlank()) {
            err += "Prenumele nu poate fi gol.\n";
        }
        if(entity.getUsername().isBlank()) {
            err += "Username-ul nu poate fi gol.\n";
        }
        if (entity.getParola().length() < 3) {
            err += "Parola trebuie sa aiba cel putin 3 caractere.\n";
        }
        if(entity.getOras().isBlank()) {
            err += "Trebuie selectat cel putin un oras.\n";
        }
        if(entity.getStrada().isBlank()) {
            err += "Strada nu poate fi goala.\n";
        }
        if(entity.getNumarStrada().isBlank()) {
            err += "Numarul strazii nu poate fi gol.\n";
        }
        if (entity.getTelefon().length() != 10) {
            err += "Numarul de telefon este invalid.\n";
        }

        if (err.length() > 0) {
            throw new Exception(err);
        }
    }
}
