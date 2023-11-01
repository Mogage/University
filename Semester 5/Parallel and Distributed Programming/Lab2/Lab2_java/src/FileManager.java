import java.io.*;
import java.util.Scanner;

public class FileManager {
    private String fileName;
    private int rowSize;
    private int columnSize;
    private int smallSize;
    private int [][] bigMatrix;
    private int [][] smallMatrix;
    private final MyFunction generateMatrix;

    public FileManager(String fileName, int rowSize, int columnSize, int smallSize, MyFunction myFunction) {
        this.fileName = fileName;
        this.rowSize = rowSize;
        this.columnSize = columnSize;
        this.smallSize = smallSize;
        this.generateMatrix = myFunction;
    }

    public int[][] getBigMatrix() {
        return bigMatrix;
    }

    public int[][] getSmallMatrix() {
        return smallMatrix;
    }

    private void writeMatrixToFile(int numRows, int numCols, int[][] matrix, PrintWriter printWriter) {
        for (int i = 0; i < numRows; i++) {
            for (int j = 0; j < numCols; j++) {
                printWriter.print(matrix[i][j] + " ");
            }
            printWriter.println();
        }
    }

    public void createFile() {
        try (FileWriter fileWriter = new FileWriter(fileName);
             PrintWriter printWriter = new PrintWriter(fileWriter)) {
            writeMatrixToFile(rowSize, columnSize, generateMatrix.apply(rowSize, columnSize, 101), printWriter);
            writeMatrixToFile(smallSize, smallSize, generateMatrix.apply(smallSize, smallSize, 2), printWriter);
        } catch (IOException e) {
            System.err.println("An error occurred while writing to the file: " + e.getMessage());
        }
    }

    public void writeMatrix(int[][] matrix) {
        try (FileWriter fileWriter = new FileWriter(fileName);
             PrintWriter printWriter = new PrintWriter(fileWriter)) {
            writeMatrixToFile(matrix.length, matrix[0].length, matrix, printWriter);
        } catch (IOException e) {
            System.err.println("An error occurred while writing to the file: " + e.getMessage());
        }
    }

    private static int[][] readMatrixFromFile(int numRows, int numCols, Scanner scanner) {
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

    public void readFile() {
        try {
            File file = new File(fileName);
            Scanner scanner = new Scanner(file);
            bigMatrix = readMatrixFromFile(rowSize, columnSize, scanner);
            smallMatrix = readMatrixFromFile(smallSize, smallSize, scanner);
            scanner.close();
        } catch (FileNotFoundException e) {
            System.err.println("File not found: " + e.getMessage());
        }
    }
}
