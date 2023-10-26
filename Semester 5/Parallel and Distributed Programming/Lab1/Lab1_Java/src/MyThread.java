
public class MyThread extends Thread {
    private int type;
    private int start;
    private int end;
    private int size;
    int[][] resultMatrix;
    private MatrixPair matrixPair;

    public MyThread(int start, int end, int type, MatrixPair matrixPair, int[][] resultMatrix) {
        this.start = start;
        this.end = end;
        this.matrixPair = matrixPair;
        this.type = type;
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

    private void computeRows() {
        this.size = matrixPair.getMatrix1()[0].length;
        for (int i = start; i < end; i++) {
            for (int j = 1; j < size - 1; j++) {
                resultMatrix[i - 1][j - 1] = computeSubMatrix(i, j);
            }
        }
    }

    private void computeCols() {
        this.size = matrixPair.getMatrix1().length;
        for (int i = start; i < end; i++) {
            for (int j = 1; j < size - 1; j++) {
                resultMatrix[j - 1][i - 1] = computeSubMatrix(j, i);
            }
        }
    }

    private void computeSubMatrices() {
        int i;
        int j;
        for (int aux = start; aux < end; aux++) {
            i = aux / (matrixPair.getMatrix1()[0].length - 2) + 1;
            j = aux % (matrixPair.getMatrix1()[0].length - 2) + 1;
            resultMatrix[i - 1][j - 1] = computeSubMatrix(i, j);
        }
    }

    @Override
    public void run() {
        if (type == 1) {
            computeRows();
        } else if (type == 2){
            computeCols();
        } else {
            computeSubMatrices();
        }
    }
}