public class Tiger implements Animal {
    private int numberOfLegs;
    private int kilogramsToEat;

    public Tiger(int numberOfLegs, int kilogramsToEat) {
        this.numberOfLegs = numberOfLegs;
        this.kilogramsToEat = kilogramsToEat;
    }

    @Override
    public Animal clone() {
        System.out.println("cloning tiger with featuers: " + this.numberOfLegs + " - " + this.kilogramsToEat);
        return new Tiger(this.numberOfLegs, this.kilogramsToEat);
    }

    @Override
    public void eat(int kilogramsEaten) {
        this.kilogramsToEat -= kilogramsEaten;
    }

    public void setNumberOfLegs(int numberOfLegs) {
        this.numberOfLegs = numberOfLegs;
    }

    public int getNumberOfLegs() {
        return numberOfLegs;
    }

    public int getKilogramsToEat() {
        return kilogramsToEat;
    }
}