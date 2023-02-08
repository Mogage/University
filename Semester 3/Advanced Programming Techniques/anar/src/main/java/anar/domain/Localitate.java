package anar.domain;

public class Localitate {
    private String nume;
    private Rau rau;
    private Integer cotaMinimaDeRisc;
    private Integer cotaMaximaAdmisa;

    public Localitate(String nume, Rau rau, Integer cotaMinimaDeRisc, Integer cotaMaximaAdmisa) {
        this.nume = nume;
        this.rau = rau;
        this.cotaMinimaDeRisc = cotaMinimaDeRisc;
        this.cotaMaximaAdmisa = cotaMaximaAdmisa;
    }

    public String getNume() {
        return nume;
    }

    public void setNume(String nume) {
        this.nume = nume;
    }

    public Rau getRau() {
        return rau;
    }

    public void setRau(Rau rau) {
        this.rau = rau;
    }

    public Integer getCotaMinimaDeRisc() {
        return cotaMinimaDeRisc;
    }

    public void setCotaMinimaDeRisc(Integer cotaMinimaDeRisc) {
        this.cotaMinimaDeRisc = cotaMinimaDeRisc;
    }

    public Integer getCotaMaximaAdmisa() {
        return cotaMaximaAdmisa;
    }

    public void setCotaMaximaAdmisa(Integer cotaMaximaAdmisa) {
        this.cotaMaximaAdmisa = cotaMaximaAdmisa;
    }
}
