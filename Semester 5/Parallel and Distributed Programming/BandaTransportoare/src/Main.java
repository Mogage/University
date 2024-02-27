import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantLock;

public class Main {
    public static int size = 100;

    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        int n = scanner.nextInt();
        int p = scanner.nextInt();
        int c = scanner.nextInt();

        AtomicInteger totalProducers = new AtomicInteger(p);
        AtomicInteger totalConsumers = new AtomicInteger(c);

        Lock lock = new ReentrantLock();
        MyQueue<Integer> banda = new MyQueue<>(lock, totalProducers, n);
        MyList<MyObject> lista = new MyList<>(lock);

        List<Thread> producers = new ArrayList<>();
        List<Thread> consumers = new ArrayList<>();

        for (int i = 0; i < p; i++) {
            producers.add(new Producer(lista, banda, totalProducers, i));
            producers.getLast().start();
        }
        for (int i = 0; i < c; i++) {
            consumers.add(new Consumer(lista, banda, totalConsumers, p + i));
            consumers.getLast().start();
        }
        Thread iterator = new ListIterator(lista, totalProducers, totalConsumers);
        iterator.start();

        for (Thread producer : producers) {
            try {
                producer.join();
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }
        for (Thread consumer : consumers) {
            try {
                consumer.join();
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }
        try {
            iterator.join();
        } catch (InterruptedException e) {
            e.printStackTrace();
        }

    }

    private static class Producer extends Thread {
        private final MyList<MyObject> list;
        private final MyQueue<Integer> banda;
        private final AtomicInteger totalProducers;
        private final int threadId;

        public Producer(MyList<MyObject> list,
                        MyQueue<Integer> banda,
                        AtomicInteger totalProducers,
                        int threadId) {
            this.list = list;
            this.banda = banda;
            this.totalProducers = totalProducers;
            this.threadId = threadId;
        }

        @Override
        public void run() {
            for (int i = 0; i < size; i++) {
                try {
                    for (int j = 0; j < 4; j++) {
                        banda.push(i, list, threadId);
                        // list.add(new MyObject(threadId, banda.size(), "push"));
                    }
                    sleep(10);
                } catch (InterruptedException e) {
                    throw new RuntimeException(e);
                }
            }
            totalProducers.decrementAndGet();
        }
    }

    private static class Consumer extends Thread {
        private final MyList<MyObject> list;
        private final MyQueue<Integer> banda;
        private final AtomicInteger totalConsumers;
        private final int threadId;

        public Consumer(MyList<MyObject> list,
                        MyQueue<Integer> banda,
                        AtomicInteger totalConsumers,
                        int threadId) {
            this.list = list;
            this.banda = banda;
            this.totalConsumers = totalConsumers;
            this.threadId = threadId;
        }

        @Override
        public void run() {
            while(banda.hasData()) {
                try {
                    for (int j = 0; j < 3; j++) {
                        banda.pop(list, threadId);
                        // list.add(new MyObject(threadId, banda.size(), "pop"));
                    }
                    sleep(8);
                } catch (InterruptedException e) {
                    throw new RuntimeException(e);
                }
            }
            totalConsumers.decrementAndGet();
        }
    }

    private static class ListIterator extends Thread {
        private final MyList<MyObject> list;
        private final AtomicInteger totalProducers;
        private final AtomicInteger totalConsumers;

        public ListIterator(MyList<MyObject> list, AtomicInteger totalProducers, AtomicInteger totalConsumers) {
            this.list = list;
            this.totalProducers = totalProducers;
            this.totalConsumers = totalConsumers;
        }

        @Override
        public void run() {
            while (totalProducers.get() != 0 || totalConsumers.get() != 0) {
                try {
                    list.printList();
                    sleep(20);
                } catch (InterruptedException e) {
                    throw new RuntimeException(e);
                }
            }
            list.printList();
        }
    }
}

