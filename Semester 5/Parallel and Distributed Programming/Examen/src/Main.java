import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;

public class Main {
    private static void generateFiles() {
        File directory = new File("files");
        if (directory.exists()) {
            for (int i = 0; i < 8; i++) {

            }
        }
    }

    public static void main(String[] args) {
        IOHandler.cleanFile();

        int F = 8;
        int start = 0;
        int end;
        int batchSize = F / 4;
        int batchReminder = F % 4;
        AtomicInteger totalProducers = new AtomicInteger(4);
        AtomicInteger totalConsumers = new AtomicInteger(8);
        AtomicInteger messagesSent = new AtomicInteger(0);
        MyQueue<Mesaj> queue = new MyQueue<>(totalProducers);
        MyMap1 map1 = new MyMap1();
        MyMap2 map2 = new MyMap2();
        List<Thread> producers = new ArrayList<>();
        List<Thread> consumers = new ArrayList<>();
        Thread supervisor;

        for (int i = 0; i < totalProducers.get(); i++) {
            end = start + batchSize;
            if (batchReminder > 0) {
                end++;
                batchReminder--;
            }
            producers.add(new Producer(queue, totalProducers, start, end, messagesSent));
            producers.getLast().start();
            start = end;
        }
        for (int i = 0; i < totalConsumers.get(); i++) {
            consumers.add(new Consumer(queue, totalConsumers, map1, map2));
            consumers.getLast().start();
        }
        supervisor = new Supervisor(queue, map1, map2, totalProducers, totalConsumers, messagesSent);
        supervisor.start();

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
            supervisor.join();
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
    }

    private static class Producer extends Thread {
        final MyQueue<Mesaj> queue;
        AtomicInteger totalProducers;
        int start;
        int end;
        AtomicInteger messagesSent;

        public Producer(MyQueue<Mesaj> queue, AtomicInteger totalProducers, int start, int end, AtomicInteger messagesSent) {
            this.queue = queue;
            this.totalProducers = totalProducers;
            this.start = start;
            this.end = end;
            this.messagesSent = messagesSent;
        }

        @Override
        public void run() {
            File directory = new File("files");

            if (directory.isDirectory()) {
                File[] files = directory.listFiles();
                if (files != null) {
                    for (int i = start; i < end; i++) {
                        File file = files[i];
                        //System.out.println("Reading file " + file.getName());
                        try (BufferedReader reader = new BufferedReader(new FileReader(file))) {
                            String line;
                            while ((line = reader.readLine()) != null) {
                                line = line.strip();
                                if (line.isEmpty()) {
                                    continue;
                                }
                                String[] values = line.split(",");
                                //System.out.println("Sending message " + Arrays.toString(values));
                                queue.push(new Mesaj(values[0], values[1], values[2]));
                                messagesSent.incrementAndGet();
                            }
                        } catch (IOException e) {
                            e.printStackTrace();
                        } catch (InterruptedException e) {
                            throw new RuntimeException(e);
                        }
                    }
                }
            }
            synchronized (this) {
                totalProducers.decrementAndGet();
                System.out.println("Producer " + start + "-" + end + " finished");
                queue.finish();
            }
        }
    }

    private static class Consumer extends Thread {
        MyQueue<Mesaj> queue;
        AtomicInteger totalConsumers;
        MyMap1 map1;
        MyMap2 map2;

        public Consumer(MyQueue<Mesaj> queue, AtomicInteger totalConsumers, MyMap1 map1, MyMap2 map2) {
            this.queue = queue;
            this.totalConsumers = totalConsumers;
            this.map1 = map1;
            this.map2 = map2;
        }

        @Override
        public void run() {
            while (queue.hasData()) {
                try {
                    Mesaj mesaj = queue.pop();
                    if (mesaj == null) {
                        continue;
                    }
                    map1.add(mesaj);
                    map2.add(mesaj);
                    sleep(2);
                } catch (InterruptedException e) {
                    throw new RuntimeException(e);
                }
            }
            totalConsumers.decrementAndGet();
            System.out.println("Consumer finished");
        }
    }

    private static class Supervisor extends Thread {
        MyQueue<Mesaj> queue;
        MyMap1 map1;
        MyMap2 map2;
        AtomicInteger totalProducers;
        AtomicInteger totalConsumers;
        AtomicInteger messagesSent;

        public Supervisor(MyQueue<Mesaj> queue,
                          MyMap1 map1,
                          MyMap2 map2,
                          AtomicInteger totalProducers,
                          AtomicInteger totalConsumers,
                          AtomicInteger messagesSent) {
            this.queue = queue;
            this.map1 = map1;
            this.map2 = map2;
            this.totalProducers = totalProducers;
            this.totalConsumers = totalConsumers;
            this.messagesSent = messagesSent;
        }

        @Override
        public void run() {
            while (totalProducers.get() > 0 || totalConsumers.get() > 0) {
                try {
                    IOHandler.writeText(LocalDateTime.now() + ": " + messagesSent.get() + "=" + queue.size() + "+" + map1.size() + "/" + map2.size() + "\n");
                    sleep(100);
                } catch (InterruptedException e) {
                    throw new RuntimeException(e);
                }
            }
            IOHandler.writeText("DONE! " + LocalDateTime.now() + ": " + messagesSent.get() + "=" + queue.size() + "+" + map1.size() + "/" + map2.size() + "\n");
        }
    }
}
