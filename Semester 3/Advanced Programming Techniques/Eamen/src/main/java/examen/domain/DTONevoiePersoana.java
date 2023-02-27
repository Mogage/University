package examen.domain;

import examen.utils.Constants;

import java.time.LocalDateTime;

public class DTONevoiePersoana extends Entity<Long>{
    private String titlu;
    private String descriere;
    private LocalDateTime deadline;
    private String omInNevoie;
    private String status;

    public DTONevoiePersoana(String titlu, String descriere, LocalDateTime deadline, String omInNevoie, String status) {
        this.titlu = titlu;
        this.descriere = descriere;
        this.deadline = deadline;
        this.omInNevoie = omInNevoie;
        this.status = status;
    }

    public String getTitlu() {
        return titlu;
    }

    public String getDescriere() {
        return descriere;
    }

    public String getDeadline() {
        return deadline.format(Constants.DATE_TIME_FORMATTER);
    }

    public String getOmInNevoie() {
        return omInNevoie;
    }

    public String getStatus() {
        return status;
    }

}
