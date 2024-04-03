public class AnimalClient {
    private final Animal animalPrototype;

    public AnimalClient(Animal animalPrototype) {
        this.animalPrototype = animalPrototype;
    }

    public Animal createAnimal() {
        return this.animalPrototype.clone();
    }
}
