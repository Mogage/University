public class Mesaj {
    String titlu_piesa;
    String nume_compozitor;
    String departament;

    public Mesaj(String titlu_piesa, String nume_compozitor, String departament) {
        this.titlu_piesa = titlu_piesa;
        this.nume_compozitor = nume_compozitor;
        this.departament = departament;
    }

    @Override
    public String toString() {
        return "Mesaj{" +
                "titlu_piesa='" + titlu_piesa + '\'' +
                ", nume_compozitor='" + nume_compozitor + '\'' +
                ", departament='" + departament + '\'' +
                '}';
    }
}
