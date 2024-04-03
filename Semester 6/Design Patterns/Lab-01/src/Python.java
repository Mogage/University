public class Python implements Animal {
    private int length;
    private int kilogramsToEat;

    public Python(int length, int kilogramsToEat) {
        this.length = length;
        this.kilogramsToEat = kilogramsToEat;
    }

    @Override
    public Animal clone() {
        return new Python(this.length, this.kilogramsToEat);
    }

    @Override
    public void eat(int kilogramsEaten) {
        this.kilogramsToEat -= kilogramsEaten;
    }

    public void setLength(int length) {
        this.length = length;
    }

    public int getLength() {
        return length;
    }

    public int getKilogramsToEat() {
        return kilogramsToEat;
    }
}
