import java.util.concurrent.CyclicBarrier;

public class MyThread extends Thread {
    private int start;
    private int end;
    private int smallSize;
    private int[][] bigMatrix;
    private int[][] smallMatrix;
    private CyclicBarrier cyclicBarrier;
    private int[][] buffer;
    private int noOfRowsBuffer;
    protected int size;

    public MyThread(int start, int end, int[][] bigMatrix, int[][] smallMatrix, CyclicBarrier cyclicBarrier) {
        this.start = start;
        this.end = end;
        this.bigMatrix = bigMatrix;
        this.smallMatrix = smallMatrix;
        this.smallSize = smallMatrix.length;
        this.cyclicBarrier = cyclicBarrier;
        this.buffer = new int[end - start + 2][bigMatrix[0].length];
        this.noOfRowsBuffer = end - start + 2;
    }

    private void createBuffer() {
        for (int i = 0; i < noOfRowsBuffer; i++) {
            if (i + start - 1 < 0 || i + start - 1 >= bigMatrix.length) {
                continue;
            }
            System.arraycopy(bigMatrix[i + start - 1], 0, buffer[i], 0, bigMatrix[0].length);
        }
        if (end == bigMatrix.length) {
            noOfRowsBuffer--;
        }
    }

    private int getIndex(int position, int index, int positionSize) {
        if (position - smallSize / 2 + index < 0)
            return 0;
        else if (position - smallSize / 2 + index >= positionSize)
            return positionSize - 1;
        else
            return position - smallSize / 2 + index;
    }

    private int computeSubMatrix(int row, int col) {
        int rowIndex;
        int columnIndex;
        int sum = 0;
        for (int index1 = 0; index1 < smallSize; index1++) {
            for (int index2 = 0; index2 < smallSize; index2++) {
                rowIndex = getIndex(row, index1, noOfRowsBuffer);
                columnIndex = getIndex(col, index2, buffer[0].length);
                sum += buffer[rowIndex][columnIndex] * smallMatrix[index1][index2];
            }
        }
        return sum;
    }

    private void computeRows() {
        this.size = bigMatrix[0].length;
        for (int i = 1; i <= end - start; i++) {
            for (int j = 0; j < size; j++) {
                bigMatrix[i + start - 1][j] = computeSubMatrix(i, j);
            }
        }
    }

    @Override
    public void run() {
        try {
            createBuffer();
            System.out.println("Threads are buffering rows " + start + " to " + end);
            cyclicBarrier.await();
            System.out.println("Threads are computing rows " + start + " to " + end);
            computeRows();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
