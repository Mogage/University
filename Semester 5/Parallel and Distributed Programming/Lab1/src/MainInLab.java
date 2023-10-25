import java.util.Arrays;
import java.util.Random;

public class MainInLab {
    private static int[] A, B, C;

    public static class MyThread2 extends Thread {
        private int id, p;

        public MyThread2(int id, int p) {
            this.id = id;
            this.p = p;
        }

        @Override
        public void run() {
            for (int i = id; i < id + p; ++i) {
                C[i] = A[i] + B[i];
            }
        }
    }

    public static void main(String[] args) {
        Random rand = new Random();
        int N = 10;
        int P = 4;
        int L = 5;
        MyThread2[] threads = new MyThread2[P];

        A = new int[N];
        B = new int[N];
        C = new int[N];

        for (int i = 0; i < A.length; i++) {
            A[i] = rand.nextInt(L) + 1;
            B[i] = rand.nextInt(L) + 1;
            C[i] = 0;
        }
        int id, pas;
        for (int i = 0; i < P; ++i) {
            id = i * P;
            if (id + P >= N) {
                pas = N - id;
            } else {
                pas = P;
            }
            threads[i] = new MyThread2(id, pas);
            threads[i].start();
        }

        for (int i = 0; i < P; ++i) {
            try {
                threads[i].join();
            } catch (InterruptedException interruptedException) {

            }
        }

        System.out.println(Arrays.toString(A));
        System.out.println(Arrays.toString(B));
        System.out.println(Arrays.toString(C));
    }

//    public static class MyThread extends Thread {
//        private int start, end;
//
//        public MyThread(int start, int end) {
//            this.start = start;
//            this.end = end;
//        }
//
//        @Override
//        public void run() {
//            for (int i = start; i < end; ++i) {
//                C[i] = A[i] + B[i];
////                C[i + 1] = A[i] + B[i];
////                if (C[i + 1] >= 10) {
////                    C[i + 1] -= 10;
////                    C[i] += 1;
////                }
//            }
//        }
//    }
//    public static void main(String[] args) {
//        Random rand = new Random();
//        int N = 10;
//        int P = 4;
//        int L = 5;
//        int threadTasks = N / P;
//        int reminder = N % P;
//        MyThread[] threads = new MyThread[P];
//
//        A = new int[N];
//        B = new int[N];
//        C = new int[N];
//
//        for (int i = 0; i < A.length; i++) {
//            A[i] = rand.nextInt(L) + 1;
//            B[i] = rand.nextInt(L) + 1;
//            C[i] = 0;
//        }
//
//        int start = 0;
//        int end;
//
//        for (int i = 0; i < P; ++i) {
//            end = start + threadTasks;
//            if (reminder > 0) {
//                --reminder;
//                ++end;
//            }
//            threads[i] = new MyThread(start, end);
//            threads[i].start();
//            start = end;
//        }
//
//        for (int i = 0; i < P; ++i) {
//            try {
//                threads[i].join();
//            } catch (InterruptedException interruptedException) {
//
//            }
//        }
//
//        System.out.println(Arrays.toString(A));
//        System.out.println(Arrays.toString(B));
//        System.out.println(Arrays.toString(C));
//    }
}