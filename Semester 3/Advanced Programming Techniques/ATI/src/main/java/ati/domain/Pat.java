package ati.domain;

public class Pat {
    private Long id;
    private String tip;
    private boolean ventilatie;
    private String cnpPacient;

    public Pat(Long id, String tip, boolean ventilatie, String cnpPacient) {
        this.id = id;
        this.tip = tip;
        this.ventilatie = ventilatie;
        this.cnpPacient = cnpPacient;
    }

    public Long getId() {
        return id;
    }

    public void setId(Long id) {
        this.id = id;
    }

    public String getTip() {
        return tip;
    }

    public void setTip(String tip) {
        this.tip = tip;
    }

    public boolean isVentilatie() {
        return ventilatie;
    }

    public void setVentilatie(boolean ventilatie) {
        this.ventilatie = ventilatie;
    }

    public String getCnpPacient() {
        return cnpPacient;
    }

    public void setCnpPacient(String cnpPacient) {
        this.cnpPacient = cnpPacient;
    }
}
