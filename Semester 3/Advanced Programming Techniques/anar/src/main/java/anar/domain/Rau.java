package anar.domain;

public class Rau {
    private String nume;
    private Integer cotaMedie;

    public Rau(String nume, Integer cotaMedie) {
        this.nume = nume;
        this.cotaMedie = cotaMedie;
    }

    public String getNume() {
        return nume;
    }

    public void setNume(String nume) {
        this.nume = nume;
    }

    public Integer getCotaMedie() {
        return cotaMedie;
    }

    public void setCotaMedie(Integer cotaMedie) {
        this.cotaMedie = cotaMedie;
    }
}
