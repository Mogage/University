import java.util.concurrent.CyclicBarrier;

public class MyThread extends Thread {
    private int start;
    private int end;
    private int smallSize;
    private int halfSmallSize;
    private int[][] bigMatrix;
    private int[][] smallMatrix;
    private CyclicBarrier cyclicBarrier;
    private int[][] buffer;
    private int noOfRowsBuffer;
    private int[] topBuffer;
    private int[] bottomBuffer;
    protected int size;

    public MyThread(int start, int end, int[][] bigMatrix, int[][] smallMatrix, CyclicBarrier cyclicBarrier) {
        this.start = start;
        this.end = end;
        this.bigMatrix = bigMatrix;
        this.smallMatrix = smallMatrix;
        this.smallSize = smallMatrix.length;
        this.halfSmallSize = smallSize / 2;
        this.cyclicBarrier = cyclicBarrier;
        this.buffer = new int[3][bigMatrix[0].length];
        this.noOfRowsBuffer = 3;
    }

    private void createBuffer() {
        if (start == 0) {
            System.arraycopy(bigMatrix[start], 0, buffer[0], 0, bigMatrix[0].length);
        } else {
            System.arraycopy(bigMatrix[start - 1], 0, buffer[0], 0, bigMatrix[0].length);
        }
        System.arraycopy(bigMatrix[start], 0, buffer[1], 0, bigMatrix[0].length);
        if (end == bigMatrix.length) {
            System.arraycopy(bigMatrix[end - 1], 0, buffer[2], 0, bigMatrix[0].length);
        } else {
            System.arraycopy(bigMatrix[end], 0, buffer[2], 0, bigMatrix[0].length);
        }
    }

    private int getIndex(int position, int index, int positionSize) {
        return Math.min(Math.max(position - halfSmallSize + index, 0), positionSize - 1);
    }

    private int computeSubMatrix(int row, int col) {
        int columnIndex;
        int sum = 0;
        for (int index1 = 0; index1 < 2; index1++) {
            for (int index2 = 0; index2 < smallSize; index2++) {
                columnIndex = getIndex(col, index2, buffer[0].length);
                sum += buffer[index1][columnIndex] * smallMatrix[index1][index2];
            }
        }

        for (int index2 = 0; index2 < smallSize; index2++) {
            columnIndex = getIndex(col, index2, buffer[0].length);
            if (row >= end - 1) {
                sum += buffer[2][columnIndex] * smallMatrix[2][index2];
                continue;
            }
            sum += bigMatrix[row + 1][columnIndex] * smallMatrix[2][index2];
        }
        return sum;
    }

    private void computeRows() {
        this.size = bigMatrix[0].length;
        for (int i = start; i < end; i++) {
            for (int j = 0; j < size; j++) {
                bigMatrix[i][j] = computeSubMatrix(i, j);
            }
            System.arraycopy(buffer[1], 0, buffer[0], 0, size);
            System.arraycopy(bigMatrix[Math.min(i + 1, end - 1)], 0, buffer[1], 0, size);
        }
    }

    @Override
    public void run() {
        try {
            createBuffer();
            cyclicBarrier.await();
            computeRows();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
