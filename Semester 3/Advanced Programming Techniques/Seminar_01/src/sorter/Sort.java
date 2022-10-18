package sorter;

public abstract class Sort {
    protected static void swap(int[] numbers, int position1, int position2) {
        int aux = numbers[position1];
        numbers[position1] = numbers[position2];
        numbers[position2] = aux;
    }
    public abstract void sort(int[] numbers, int start, int size) throws Exception;
}
