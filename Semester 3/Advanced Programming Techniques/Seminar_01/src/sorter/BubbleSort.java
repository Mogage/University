package sorter;

public class BubbleSort extends Sort {
    public void sort(int[] numbers, int start, int size) throws Exception {
        if ( size > numbers.length) {
            throw new Exception("Size given greater than array length.\n");
        }

        for (int count1 = start; count1 < size; count1++) {
            for (int count2 = start; count2 < size; count2++) {
                if (numbers[count1] < numbers[count2]) {
                    swap(numbers, count1, count2);
                }
            }
        }
    }
}
