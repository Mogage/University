public class Main {
    public static void main(String[] args) {
        Animal animalPrototype1 = new Tiger(4, 50);
        Animal animalPrototype2 = new Tiger(4, 200);
        AnimalClient tigerClient1 = new AnimalClient(animalPrototype1);
        AnimalClient tigerClient2 = new AnimalClient(animalPrototype2);

        Animal babyTiger = tigerClient1.createAnimal();
        Animal adultTiger = tigerClient2.createAnimal();

        babyTiger.eat(5);
        adultTiger.eat(15);


    }
}