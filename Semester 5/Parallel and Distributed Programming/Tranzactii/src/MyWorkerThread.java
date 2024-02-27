import java.util.Random;
import java.util.concurrent.atomic.AtomicInteger;

public class MyWorkerThread extends Thread {
    final MyDictionary dictionary;
    final Integer size;
    final String client;
    final String operationType;

    public MyWorkerThread(MyDictionary dictionary, Integer size, String client, String operationType) {
        this.dictionary = dictionary;
        this.size = size;
        this.client = client;
        this.operationType = operationType;
    }

    @Override
    public void run() {
        IOHandler.writeText("Thread " + threadId() + " started, verifying if " + client + " is a valid key\n");
        dictionary.lock.lock();
            if (!dictionary.verifyKey(client)) {
                IOHandler.writeText("Key " + client + " doesn't exist, Thread " + threadId() + " inserting it\n");
                dictionary.insertKey(client);
            }
        dictionary.lock.unlock();
        Random random = new Random();
        for (int i = 0; i < size; i++) {
            int value = random.nextInt(10) + 1;
            dictionary.pushValue(client, operationType, value);
        }
        IOHandler.writeText("Thread " + threadId() + " done\n");
        System.out.println("Thread " + threadId() + " done");
    }
}

