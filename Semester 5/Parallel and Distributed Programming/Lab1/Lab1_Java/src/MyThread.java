
public class MyThread extends Thread {
    private boolean onRows;
    private int start;
    private int end;
    private int size;
    int[][] resultMatrix;
    private MatrixPair matrixPair;

    public MyThread(int start, int end, boolean onRows, MatrixPair matrixPair, int[][] resultMatrix) {
        this.start = start;
        this.end = end;
        this.matrixPair = matrixPair;
        this.onRows = onRows;
        this.resultMatrix = resultMatrix;
    }

    private int computeSubMatrix(int row, int col) {
        int sum = 0;
        for (int index1 = row - 1; index1 <= row + 1; index1++) {
            for (int index2 = col - 1; index2 <= col + 1; index2++) {
                sum += matrixPair.getMatrix1()[index1][index2] * matrixPair.getMatrix2()[index1 - row + 1][index2 - col + 1];
            }
        }
        return sum;
    }

    private void computeMatrix() {
        for (int i = start; i < end; i++) {
            for (int j = 1; j < size - 1; j++) {
                resultMatrix[i - 1][j - 1] = computeSubMatrix(i, j);
            }
        }
    }

    private void computeRows() {
        this.size = matrixPair.getMatrix1()[0].length;
        computeMatrix();
    }

    private void computeCols() {
        this.size = matrixPair.getMatrix1().length;
        computeMatrix();
    }

    @Override
    public void run() {
        if (onRows) {
            computeRows();
        } else {
            computeCols();
        }
    }
}