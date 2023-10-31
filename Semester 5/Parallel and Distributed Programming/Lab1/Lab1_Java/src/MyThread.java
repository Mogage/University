
public class MyThread extends Thread {
    private int type;
    private int start;
    private int end;
    private int size;
    private int smallSize;
    int[][] resultMatrix;
    private MatrixPair matrixPair;

    public MyThread(int start, int end, int type, MatrixPair matrixPair, int[][] resultMatrix) {
        this.start = start;
        this.end = end;
        this.matrixPair = matrixPair;
        this.type = type;
        this.resultMatrix = resultMatrix;
        this.smallSize = matrixPair.getMatrix2().length;
    }

    private int computeSubMatrix(int row, int col) {
        int rowIndex;
        int columnIndex;
        int sum = 0;
        for (int index1 = 0; index1 < smallSize; index1++) {
            for (int index2 = 0; index2 < smallSize; index2++) {
                if (row - smallSize / 2 + index1 < 0)
                    rowIndex = 0;
                else if (row - smallSize / 2 + index1 >= matrixPair.getMatrix1().length)
                    rowIndex = matrixPair.getMatrix1().length - 1;
                else
                    rowIndex = row - smallSize / 2 + index1;
                if (col - smallSize / 2 + index2 < 0)
                    columnIndex = 0;
                else if (col - smallSize / 2 + index2 >= matrixPair.getMatrix1()[0].length)
                    columnIndex = matrixPair.getMatrix1()[0].length - 1;
                else
                    columnIndex = col - smallSize / 2 + index2;
                sum += matrixPair.getMatrix1()[rowIndex][columnIndex] * matrixPair.getMatrix2()[index1][index2];
            }
        }
        return sum;
    }

    private void computeRows() {
        this.size = matrixPair.getMatrix1()[0].length;
        for (int i = start; i < end; i++) {
            for (int j = 0; j < size; j++) {
                resultMatrix[i][j] = computeSubMatrix(i, j);
            }
        }
    }

    private void computeCols() {
        this.size = matrixPair.getMatrix1().length;
        for (int i = start; i < end; i++) {
            for (int j = 0; j < size; j++) {
                resultMatrix[j][i] = computeSubMatrix(j, i);
            }
        }
    }

    private void computeSubMatrices() {
        int i;
        int j;
        for (int aux = start; aux < end; aux++) {
            i = aux / matrixPair.getMatrix1()[0].length;
            j = aux % matrixPair.getMatrix1()[0].length;
            resultMatrix[i][j] = computeSubMatrix(i, j);
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