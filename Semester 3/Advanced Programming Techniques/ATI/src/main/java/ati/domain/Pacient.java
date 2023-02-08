package ati.domain;

public class Pacient {
    private String cnp;
    private int varsta;
    private boolean prematur;
    private String diagnostic;
    private Integer gravitate;

    public Pacient(String cnp, int varsta, boolean prematur, String diagnostic, int gravitate) {
        this.cnp = cnp;
        this.varsta = varsta;
        this.prematur = prematur;
        this.diagnostic = diagnostic;
        this.gravitate = gravitate;
    }

    public String getCnp() {
        return cnp;
    }

    public void setCnp(String cnp) {
        this.cnp = cnp;
    }

    public int getVarsta() {
        return varsta;
    }

    public void setVarsta(int varsta) {
        this.varsta = varsta;
    }

    public boolean isPrematur() {
        return prematur;
    }

    public void setPrematur(boolean prematur) {
        this.prematur = prematur;
    }

    public String getDiagnostic() {
        return diagnostic;
    }

    public void setDiagnostic(String diagnostic) {
        this.diagnostic = diagnostic;
    }

    public Integer getGravitate() {
        return gravitate;
    }

    public void setGravitate(int gravitate) {
        this.gravitate = gravitate;
    }
}
