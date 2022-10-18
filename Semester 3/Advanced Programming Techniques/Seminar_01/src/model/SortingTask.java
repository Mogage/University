package model;

import sorter.Sort;

public class SortingTask extends Task{
    private final int[] numbers;

    private final Sort sorter;

    public SortingTask(String _taskId, String _description, int[] numbers, Sort sort) {
        super(_taskId, _description);
        this.numbers = numbers;
        this.sorter = sort;
    }

    private void printArray() {
        for(int number : numbers){
            System.out.print(number);
            System.out.print("  ");
        }
        System.out.println();
    }

    public void execute() {
        try {
            sorter.sort(numbers, 0, numbers.length);
        }
        catch (Exception exception){
            exception.printStackTrace();
        }
        printArray();
    }
}
