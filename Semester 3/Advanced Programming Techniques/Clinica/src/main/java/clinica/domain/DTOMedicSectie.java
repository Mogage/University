package clinica.domain;

public class DTOMedicSectie {
    private final String numeSectie;
    private final String numeSefDeSectie;
    private final int pret;

    public DTOMedicSectie(String numeSectie, String numeSefDeSectie, int pret) {
        this.numeSectie = numeSectie;
        this.numeSefDeSectie = numeSefDeSectie;
        this.pret = pret;
    }

    public String getNumeSectie() {
        return numeSectie;
    }

    public String getNumeSefDeSectie() {
        return numeSefDeSectie;
    }

    public int getPret() {
        return pret;
    }
}
