package examen.domain;

public class Persoana extends Entity<Long> {
    private String nume;
    private String prenume;
    private String username;
    private String parola;
    private String oras;
    private String strada;
    private String numarStrada;
    private String telefon;

    public Persoana(String nume, String prenume, String username, String parola, String oras, String strada, String numarStrada, String telefon) {
        this.nume = nume;
        this.prenume = prenume;
        this.username = username;
        this.parola = parola;
        this.oras = oras;
        this.strada = strada;
        this.numarStrada = numarStrada;
        this.telefon = telefon;
    }

    public String getNume() {
        return nume;
    }

    public String getPrenume() {
        return prenume;
    }

    public String getUsername() {
        return username;
    }

    public String getParola() {
        return parola;
    }

    public String getOras() {
        return oras;
    }

    public String getStrada() {
        return strada;
    }

    public String getNumarStrada() {
        return numarStrada;
    }

    public String getTelefon() {
        return telefon;
    }
}
