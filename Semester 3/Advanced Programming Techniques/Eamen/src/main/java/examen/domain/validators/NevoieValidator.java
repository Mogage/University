package examen.domain.validators;

import examen.domain.Nevoie;

public class NevoieValidator implements Validator<Nevoie> {
    private static final NevoieValidator instance = new NevoieValidator();

    private NevoieValidator() {}

    public static NevoieValidator getInstance() {
        return instance;
    }
    
    @Override
    public void validate(Nevoie entity) throws Exception {
        String err = "";

        if (entity.getTitlu().isBlank()) {
            err += "Titlul nu poate fi gol.\n";
        }
        if(entity.getDescriere().isBlank()) {
            err += "Descrierea nu poate fi goala.\n";
        }

        if (err.length() > 0) {
            throw new Exception(err);
        }
    }
}
