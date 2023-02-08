package clinica.domain;

public class DTOMedicSectie {
    private final Long id;
    private final String numeSectie;
    private final String numeSefDeSectie;
    private final int pret;

    public DTOMedicSectie(Long id, String numeSectie, String numeSefDeSectie, int pret) {
        this.id = id;
        this.numeSectie = numeSectie;
        this.numeSefDeSectie = numeSefDeSectie;
        this.pret = pret;
    }

    public Long getId() {
        return id;
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
