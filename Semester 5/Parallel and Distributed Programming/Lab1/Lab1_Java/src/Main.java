import java.io.*;
import java.util.*;

public class Main {
    static int rowSize = 10000;
    static int columnSize = 10;
    static int smallSize = 5;
    static int[][] resultMatrix = new int[rowSize][columnSize];

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

    public static void writeMatrixToFile(int numRows, int numCols, int[][] matrix, PrintWriter printWriter) {
        for (int i = 0; i < numRows; i++) {
            for (int j = 0; j < numCols; j++) {
                printWriter.print(matrix[i][j] + " ");
            }
            printWriter.println();
        }
    }

    public static void createFile(String filename) {
        try (FileWriter fileWriter = new FileWriter(filename);
             PrintWriter printWriter = new PrintWriter(fileWriter)) {
            writeMatrixToFile(rowSize, columnSize, generateMatrix(rowSize, columnSize, 101), printWriter);
            writeMatrixToFile(smallSize, smallSize, generateMatrix(smallSize, smallSize, 2), printWriter);
        } catch (IOException e) {
            System.err.println("An error occurred while writing to the file: " + e.getMessage());
        }
    }

    public static int[][] readMatrixFromFile(int numRows, int numCols, Scanner scanner) {
        int[][] matrix = new int[numRows][numCols];

        for (int row = 0; row < numRows; row++) {
            String line = scanner.nextLine();
            String[] values = line.split(" ");
            for (int col = 0; col < numCols; col++) {
                matrix[row][col] = Integer.parseInt(values[col]);
            }
        }

        return matrix;
    }

    public static MatrixPair readFile(String filename) {
        try {
            File file = new File(filename);
            Scanner scanner = new Scanner(file);
            int[][] bigMatrix = readMatrixFromFile(rowSize, columnSize, scanner);
            int[][] smallMatrix = readMatrixFromFile(smallSize, smallSize, scanner);
            scanner.close();
            return new MatrixPair(bigMatrix, smallMatrix);
        } catch (FileNotFoundException e) {
            System.err.println("File not found: " + e.getMessage());
        }
        return null;
    }

    public static void createBiggerMatrix(MatrixPair matrixPair) {
        int noRows = matrixPair.getMatrix1().length;
        int noCols = matrixPair.getMatrix1()[0].length;
        int[][] biggerMatrix = new int[noRows + 2][noCols + 2];

        for (int i = 0; i < noRows; i++) {
            biggerMatrix[i + 1][0] = matrixPair.getMatrix1()[i][0];
            biggerMatrix[i + 1][noCols + 1] = matrixPair.getMatrix1()[i][noCols - 1];

            for (int j = 0; j < noCols; j++) {
                biggerMatrix[i + 1][j + 1] = matrixPair.getMatrix1()[i][j];
            }
        }
        for (int i = 0; i < noCols; i++) {
            biggerMatrix[0][i + 1] = matrixPair.getMatrix1()[0][i];
            biggerMatrix[noRows + 1][i + 1] = matrixPair.getMatrix1()[noRows - 1][i];
        }
        biggerMatrix[0][0] = matrixPair.getMatrix1()[0][0];
        biggerMatrix[0][noCols + 1] = matrixPair.getMatrix1()[0][noCols - 1];
        biggerMatrix[noRows + 1][0] = matrixPair.getMatrix1()[noRows - 1][0];
        biggerMatrix[noRows + 1][noCols + 1] = matrixPair.getMatrix1()[noRows - 1][noCols - 1];

        matrixPair.setMatrix1(biggerMatrix);
    }

    public static void sequential(MatrixPair matrixPair) {
        int rowIndex;
        int columnIndex;
        int sum;
        for (int i = 0; i < rowSize; i++) {
            for (int j = 0; j < columnSize; j++) {
                sum = 0;
                for (int index1 = 0; index1 < smallSize; index1++) {
                    for (int index2 = 0; index2 < smallSize; index2++) {
                        if (i - smallSize / 2 + index1 < 0)
                            rowIndex = 0;
                        else if (i - smallSize / 2 + index1 >= matrixPair.getMatrix1().length)
                            rowIndex = matrixPair.getMatrix1().length - 1;
                        else
                            rowIndex = i - smallSize / 2 + index1;
                        if (j - smallSize / 2 + index2 < 0)
                            columnIndex = 0;
                        else if (j - smallSize / 2 + index2 >= matrixPair.getMatrix1()[0].length)
                            columnIndex = matrixPair.getMatrix1()[0].length - 1;
                        else
                            columnIndex = j - smallSize / 2 + index2;
                        sum += matrixPair.getMatrix1()[rowIndex][columnIndex] * matrixPair.getMatrix2()[index1][index2];
                    }
                }
                resultMatrix[i][j] = sum;
            }
        }
    }

    public static void multiThread(MatrixPair matrixPair, int size, int type, int noOfCores, int startValue) {
        int batchSize = size / noOfCores;
        int batchReminder = size % noOfCores;
        int start = startValue;
        int end;
        MyThread[] threads = new MyThread[noOfCores + 1];

        for (int i = 0; i < noOfCores; i++) {
            end = start + batchSize;
            if (batchReminder > 0) {
                --batchReminder;
                ++end;
            }
            threads[i] = new MyThread(start, end, type, matrixPair, resultMatrix);
            threads[i].start();
            start = end;
        }

        for (int i = 0; i < noOfCores; i++) {
            try {
                threads[i].join();
            } catch (InterruptedException interruptedException) {
            }
        }
    }

    public static void printMatrix(int[][] matrix) {
        int noCols = matrix[0].length;

        for (int[] ints : matrix) {
            for (int j = 0; j < noCols; ++j) {
                System.out.print(ints[j] + " ");
            }
            System.out.println();
        }
        System.out.println();
    }

    private static double runScript(String[] args, String filename) {
        MatrixPair matrixPair = readFile(filename);
//        createBiggerMatrix(matrixPair);
        long startTime;
        if (Objects.equals(args[1], "sec")) {
            startTime = System.nanoTime();
            sequential(matrixPair);
        }
        else if (Objects.equals(args[1], "row")) {
            startTime = System.nanoTime();
            multiThread(matrixPair, rowSize, 1, Integer.parseInt(args[0]), 0);
        }
        else if (Objects.equals(args[1], "col")){
            startTime = System.nanoTime();
            multiThread(matrixPair, columnSize, 2, Integer.parseInt(args[0]), 0);
        }
        else {
            startTime = System.nanoTime();
            multiThread(matrixPair, rowSize * columnSize, 3, Integer.parseInt(args[0]), 0);
        }
        long endTime = System.nanoTime();
        return (double)(endTime - startTime)/1E6;
//        printMatrix(resultMatrix);
    }

    private static void compareResults(String type) throws Exception {
        try {
            File secFile = new File("output-sec.txt");
            Scanner secScanner = new Scanner(secFile);
            File resultFile = new File("output-" + type + ".txt");
            Scanner resultScanner = new Scanner(resultFile);
            int[][] resultMatrix = readMatrixFromFile(rowSize, columnSize, resultScanner);
            int[][] secMatrix = readMatrixFromFile(rowSize, columnSize, secScanner);
            secScanner.close();
            resultScanner.close();
            for (int i = 0; i < rowSize; i++) {
                for (int j = 0; j < columnSize; j++) {
                    if (resultMatrix[i][j] != secMatrix[i][j]) {
                        throw new Exception("Wrong result");
                    }
                }
            }
        } catch (FileNotFoundException e) {
            System.err.println("File not found: " + e.getMessage());
        }
    }

    private static void runProgram(String[] args, String filename) throws Exception {
        double time = runScript(args, filename);

        try (FileWriter fileWriter = new FileWriter("output-" + args[1] + ".txt");
             PrintWriter printWriter = new PrintWriter(fileWriter)) {
            writeMatrixToFile(rowSize, columnSize, resultMatrix, printWriter);
        } catch (IOException e) {
            System.err.println("An error occurred while writing to the file: " + e.getMessage());
        }

        if (!Objects.equals(args[1], "sec")) {
            compareResults(args[1]);
        }

        System.out.println(time);
    }

    public static void main(String[] args) throws Exception {
        String filename = "data.txt";
//        createFile(filename);
        runProgram(args, filename);
    }
}