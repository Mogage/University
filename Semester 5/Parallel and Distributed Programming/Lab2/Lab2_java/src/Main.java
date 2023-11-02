import java.io.File;
import java.io.FileNotFoundException;
import java.util.Objects;
import java.util.Random;
import java.util.Scanner;
import java.util.concurrent.CyclicBarrier;

public class Main {
    private static String fileName = "data.txt";
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
                interruptedException.printStackTrace();
            }
        }
    }

    public static void sequential(int[][] bigMatrix, int[][] smallMatrix) {
        int rowIndex;
        int columnIndex;
        int sum;
        int rowSize = bigMatrix.length;
        int columnSize = bigMatrix[0].length;
        int smallSize = smallMatrix.length;
        int noOfRowsBuffer = smallSize / 2 + 1;
        int[][] buffer = new int[noOfRowsBuffer][columnSize];

        for (int i = 0; i < noOfRowsBuffer; i++) {
            System.arraycopy(bigMatrix[0], 0, buffer[i], 0, columnSize);
        }

        for (int i = 0; i < rowSize - 1; i++) {
//            for (int k = 0; k < noOfRowsBuffer; k++) {
//                    for (int j = 0; j < buffer[0].length; j++) {
//                        System.out.print(buffer[k][j] + " ");
//                    }
//                    System.out.println();
//                }
            for (int j = 0; j < columnSize; j++) {
                sum = 0;
                for (int index1 = 0; index1 < noOfRowsBuffer; index1++) {
                    for (int index2 = 0; index2 < smallSize; index2++) {
                        columnIndex = Math.min(Math.max(j - smallSize / 2 + index2, 0), columnSize - 1);
                        sum += buffer[index1][columnIndex] * smallMatrix[index1][index2];
                    }
                }
                for (int index1 = noOfRowsBuffer; index1 < smallSize; index1++) {
                    for (int index2 = 0; index2 < smallSize; index2++) {
                        rowIndex = Math.min(Math.max(i - smallSize / 2 + index1, 0), rowSize - 1);
                        columnIndex = Math.min(Math.max(j - smallSize / 2 + index2, 0), columnSize - 1);
                        sum += bigMatrix[rowIndex][columnIndex] * smallMatrix[index1][index2];
                    }
                }
                bigMatrix[i][j] = sum;
            }
            for (int index1 = 0; index1 < noOfRowsBuffer - 1; index1++) {
                System.arraycopy(buffer[index1 + 1], 0, buffer[index1], 0, columnSize);
            }
            System.arraycopy(bigMatrix[i + 1], 0, buffer[noOfRowsBuffer - 1], 0, columnSize);
        }
        for (int j = 0; j < columnSize; j++) {
            sum = 0;
            for (int index1 = 0; index1 < smallSize; index1++) {
                for (int index2 = 0; index2 < smallSize; index2++) {
                    rowIndex = Math.min(index1, smallSize / 2);
                    columnIndex = Math.min(Math.max(j - smallSize / 2 + index2, 0), columnSize - 1);
                    sum += buffer[rowIndex][columnIndex] * smallMatrix[index1][index2];
                }
            }
            bigMatrix[rowSize - 1][j] = sum;
        }
    }

    public static void compareResults(FileManager fileManager, String fileName1, String fileName2, int rowSize, int columnSize) {
        try {
            File secFile = new File(fileName1);
            Scanner secScanner = new Scanner(secFile);
            File resultFile = new File(fileName2);
            Scanner resultScanner = new Scanner(resultFile);
            int[][] resultMatrix = fileManager.readMatrixFromFile(rowSize, columnSize, resultScanner);
            int[][] secMatrix = fileManager.readMatrixFromFile(rowSize, columnSize, secScanner);
            secScanner.close();
            resultScanner.close();
            for (int i = 0; i < rowSize; i++) {
                for (int j = 0; j < columnSize; j++) {
                    if (resultMatrix[i][j] != secMatrix[i][j]) {
                        System.out.println("Wrong result at position: " + i + " " + j);
                        return;
                    }
                }
            }
        } catch (FileNotFoundException e) {
            System.err.println("File not found: " + e.getMessage());
        }
    }

    public static void runProgram(FileManager fileManager, String[] args) {
        fileManager.readFile(fileName);
        int[][] bigMatrix = fileManager.getBigMatrix();
        int[][] smallMatrix = fileManager.getSmallMatrix();

        long startTime;
        if (Objects.equals(args[1], "sec")) {
            startTime = System.nanoTime();
            sequential(bigMatrix, smallMatrix);
        } else {
            startTime = System.nanoTime();
            multiThread(bigMatrix, smallMatrix, bigMatrix.length, 4);
        }
        long endTime = System.nanoTime();
        System.out.println("Time: " + (endTime - startTime) / 1e6 + " ms");

        fileManager.writeMatrix(bigMatrix, "result-" + args[1] + ".txt");

        if (!Objects.equals(args[1], "sec")) {
            compareResults(fileManager, "result-sec.txt", "result-row.txt", bigMatrix.length, bigMatrix[0].length);
        }
    }

    public static void main(String[] args) {
        FileManager fileManager = new FileManager(1000, 1000, 5, Main::generateMatrix);

        if (Objects.equals(args[0], "1")) {
            fileManager.createFile(fileName);
            return;
        }
        runProgram(fileManager, args);
    }
}