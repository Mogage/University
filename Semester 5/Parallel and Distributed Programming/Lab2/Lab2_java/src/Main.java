import java.util.Random;
import java.util.concurrent.CyclicBarrier;

public class Main {
    public static int[][] generateMatrix(int numRows, int numCols, int bound) {
        int[][] matrix = new int[numRows][numCols];
        Random random = new Random();

        for (int i = 0; i < numRows; i++) {
            for (int j = 0; j < numCols; j++) {
                matrix[i][j] = random.nextInt(bound);
            }
        }

        return matrix;
    }

    public static void multiThread(int[][] bigMatrix, int[][] smallMatrix, int size, int noOfThreads) {
        int batchSize = size / noOfThreads;
        int batchReminder = size % noOfThreads;
        int start = 0;
        int end;
        CyclicBarrier cyclicBarrier = new CyclicBarrier(noOfThreads);
        MyThread[] threads = new MyThread[noOfThreads];

        for (int i = 0; i < noOfThreads; i++) {
            end = start + batchSize;
            if (batchReminder > 0) {
                --batchReminder;
                ++end;
            }
            threads[i] = new MyThread(start, end, bigMatrix, smallMatrix, cyclicBarrier);
            threads[i].start();
            start = end;
        }

        for (int i = 0; i < noOfThreads; i++) {
            try {
                threads[i].join();
            } catch (InterruptedException interruptedException) {
            }
        }
    }

    private static int computeSubMatrix(int row, int col, int[][] bigMatrix, int[][] smallMatrix) {
        return 0;
    }

    public static void sequential(int[][] bigMatrix, int[][] smallMatrix) {
        int rowIndex;
        int columnIndex;
        int sum;
        int rowSize = bigMatrix.length;
        int columnSize = bigMatrix[0].length;
        int smallSize = smallMatrix.length;
        int noOfRowsBuffer = smallSize / 2 + 1;
        int[][] usedBuffer = new int[noOfRowsBuffer][columnSize];

//        for (int i = 0; i < noOfRowsBuffer; i++) {
//            System.arraycopy(bigMatrix[i], 0, usedBuffer[i], 0, columnSize);
//        }

        System.arraycopy(bigMatrix[0], 0, usedBuffer[noOfRowsBuffer - 1], 0, columnSize);

        for (int i = 0; i < rowSize; i++) {

            System.out.println(i);
            for (int[] row : usedBuffer) {
                for (int col : row) {
                    System.out.print(col + " ");
                }
                System.out.println();
            }
            System.out.println();

            for (int j = 0; j < columnSize; j++) {
                sum = 0;
                for (int index1 = 0; index1 < noOfRowsBuffer; index1++) {
                    for (int index2 = 0; index2 < smallSize; index2++) {
                        rowIndex = Math.max(index1 - smallSize / 2, 0);
                        if (j - smallSize / 2 + index2 < 0)
                            columnIndex = 0;
                        else if (j - smallSize / 2 + index2 >= columnSize)
                            columnIndex = columnSize - 1;
                        else
                            columnIndex = j - smallSize / 2 + index2;
                        sum += usedBuffer[rowIndex][columnIndex] * smallMatrix[index1][index2];
                    }
                }
                for (int index1 = noOfRowsBuffer; index1 < smallSize; index1++) {
                    for (int index2 = 0; index2 < smallSize; index2++) {
                        if (i - smallSize / 2 + index1 < 0)
                            rowIndex = 0;
                        else if (i - smallSize / 2 + index1 >= rowSize)
                            rowIndex = rowSize - 1;
                        else
                            rowIndex = i - smallSize / 2 + index1;
                        if (j - smallSize / 2 + index2 < 0)
                            columnIndex = 0;
                        else if (j - smallSize / 2 + index2 >= columnSize)
                            columnIndex = columnSize - 1;
                        else
                            columnIndex = j - smallSize / 2 + index2;
                        sum += bigMatrix[rowIndex][columnIndex] * smallMatrix[index1][index2];
                    }
                }
                bigMatrix[i][j] = sum;
            }
            if (i < rowSize - 1) {
                for (int index1 = 0; index1 < noOfRowsBuffer - 1; index1++) {
                    System.arraycopy(usedBuffer[index1 + 1], 0, usedBuffer[index1], 0, columnSize);
                }
                System.arraycopy(bigMatrix[i + 1], 0, usedBuffer[noOfRowsBuffer - 1], 0, columnSize);
            }
            for (int[] row : bigMatrix) {
                for (int col : row) {
                    System.out.print(col + " ");
                }
                System.out.println();
            }
            System.out.println();
        }

    }

    public static void runProgram(FileManager fileManager) {
        fileManager.readFile();
        int[][] bigMatrix = fileManager.getBigMatrix();
        int[][] smallMatrix = fileManager.getSmallMatrix();
        int size = bigMatrix.length;
        int noOfThreads = 4;

        long startTime = System.nanoTime();
        multiThread(bigMatrix, smallMatrix, size, noOfThreads);
        long endTime = System.nanoTime();
        System.out.println("Time: " + (endTime - startTime) / 1e6 + " ms");

        for (int[] row : bigMatrix) {
            for (int col : row) {
                System.out.print(col + " ");
            }
            System.out.println();
        }
    }

    public static void main(String[] args) {
        String fileName = "data.txt";
        FileManager fileManager = new FileManager(fileName, 10, 10, 3, Main::generateMatrix);
//        fileManager.createFile();
//        runProgram(fileManager);

        fileManager.readFile();
        sequential(fileManager.getBigMatrix(), fileManager.getSmallMatrix());
    }
}