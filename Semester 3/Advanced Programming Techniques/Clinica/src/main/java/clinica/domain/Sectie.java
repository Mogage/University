package clinica.domain;

public class Sectie extends Entity<Long> {
    private String nume;

    private Long idSefDeSectie;

    private int pretPerConsultatie;

    private int durataMaximaConsultatie;

    public Sectie(String nume, Long idSefDeSectie, int pretPerConsultatie, int durataMaximaConsultatie) {
        this.nume = nume;
        this.idSefDeSectie = idSefDeSectie;
        this.pretPerConsultatie = pretPerConsultatie;
        this.durataMaximaConsultatie = durataMaximaConsultatie;
    }

    public String getNume() {
        return nume;
    }

    public void setNume(String nume) {
        this.nume = nume;
    }

    public Long getIdSefDeSectie() {
        return idSefDeSectie;
    }

    public void setIdSefDeSectie(Long idSefDeSectie) {
        this.idSefDeSectie = idSefDeSectie;
    }

    public int getPretPerConsultatie() {
        return pretPerConsultatie;
    }

    public void setPretPerConsultatie(int pretPerConsultatie) {
        this.pretPerConsultatie = pretPerConsultatie;
    }

    public int getDurataMaximaConsultatie() {
        return durataMaximaConsultatie;
    }

    public void setDurataMaximaConsultatie(int durataMaximaConsultatie) {
        this.durataMaximaConsultatie = durataMaximaConsultatie;
    }
}
