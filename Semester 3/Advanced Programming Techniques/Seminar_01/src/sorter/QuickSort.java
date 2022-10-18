package sorter;

public class QuickSort extends Sort {
    private int partition(int[] numbers, int low, int high) {
        int pivot = numbers[high];

        int i = (low - 1);

        for (int j = low; j <= high - 1; j++) {
            if (numbers[j] <= pivot) {
                i++;
                swap(numbers, i, j);
            }
        }
        swap(numbers, i + 1, high);
        return (i + 1);
    }

    public void sort(int[] numbers, int start, int size) throws Exception {
        if (size >= start) {
            return;
        }
        int pivot = partition(numbers, start, size);
        sort(numbers, start, pivot);
        sort(numbers, pivot + 1, size);
    }
}
