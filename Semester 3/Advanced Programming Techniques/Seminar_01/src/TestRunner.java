import model.MessageTask;
import model.SortingTask;
import model.Task;
import sorter.BubbleSort;
import sorter.QuickSort;
import taskRunner.DelayTaskRunner;
import taskRunner.PrinterTaskRunner;
import taskRunner.StrategyTaskRunner;
import taskRunner.TaskRunner;
import utils.Constants;

import java.time.LocalDateTime;

public class TestRunner {
    private static MessageTask[] createMessages() {
        MessageTask m1 = new MessageTask("id1",
                "Tema map",
                "done",
                "ubb",
                "nicu",
                LocalDateTime.now());
        MessageTask m2 = new MessageTask("id2",
                "Examen map",
                "Sesiune",
                "ubb",
                "denis",
                LocalDateTime.now());
        MessageTask m3 = new MessageTask("id3",
                "Craciun",
                "Vacanta",
                "daria",
                "nicu",
                LocalDateTime.now());
        return new MessageTask[]{m1, m2, m3};
    }

    private static void runMessageTask() {
        MessageTask[] messages = createMessages();
        for (MessageTask message : messages) {
            message.execute();
        }
    }

    private static void runSortingTask() {
        BubbleSort bubbleSort = new BubbleSort();
        QuickSort quickSort = new QuickSort();
        int[] numberToSort = new int[]{6, 5, 2, 10, 654, 24, 1, 0, 43, 7, 5};
        SortingTask sortingTask1 = new SortingTask("task id 1", "sorteaza cu bubble", numberToSort, bubbleSort);
        SortingTask sortingTask2 = new SortingTask("task id 2", "sorteaza cu quick", numberToSort, quickSort);
        sortingTask1.execute();
        sortingTask2.execute();
    }

    private static void runStrategyTaskRunner(MessageTask[] messageTasks, TaskRunner taskRunner) {
        assert (!taskRunner.hasTask());
        for (MessageTask messageTask : messageTasks) {
            taskRunner.addTask(messageTask);
        }
        assert (taskRunner.hasTask());
        taskRunner.executeOneTask();
        assert (taskRunner.hasTask());
        taskRunner.executeAll();
        assert (!taskRunner.hasTask());
    }

    private static void runTaskRunners() {
        MessageTask[] messageTasks = createMessages();
        TaskRunner strategyTaskRunner = new StrategyTaskRunner(Constants.Strategy.LIFO);
        TaskRunner strategyTaskRunnerFifo = new StrategyTaskRunner(Constants.Strategy.FIFO);
        TaskRunner printerTaskRunner = new PrinterTaskRunner(strategyTaskRunner);
        TaskRunner delayTaskRunner = new DelayTaskRunner(strategyTaskRunner);
        System.out.println("Strategy LIFO task runner: ");
        runStrategyTaskRunner(messageTasks, strategyTaskRunner);
        System.out.println("Strategy FIFO task runner: ");
        runStrategyTaskRunner(messageTasks, strategyTaskRunnerFifo);
        System.out.println("Printer task runner: ");
        runStrategyTaskRunner(messageTasks, printerTaskRunner);
        System.out.println("Delay task runner: ");
        runStrategyTaskRunner(messageTasks, delayTaskRunner);
    }

    public static void run() {
        runMessageTask();
        runSortingTask();
        runTaskRunners();
    }
}









