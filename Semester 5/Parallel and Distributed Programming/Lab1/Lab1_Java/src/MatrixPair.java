public class MatrixPair {
    private int[][] matrix1;
    private int[][] matrix2;

    public MatrixPair(int[][] matrix1, int[][] matrix2) {
        this.matrix1 = matrix1;
        this.matrix2 = matrix2;
    }

    public int[][] getMatrix1() {
        return matrix1;
    }

    public int[][] getMatrix2() {
        return matrix2;
    }

    public void setMatrix1(int[][] matrix1) {
        this.matrix1 = matrix1;
    }

    public void setMatrix2(int[][] matrix2) {
        this.matrix2 = matrix2;
    }
}