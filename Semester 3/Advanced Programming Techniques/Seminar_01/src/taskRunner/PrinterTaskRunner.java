package taskRunner;

import utils.Constants;

import java.time.LocalDateTime;

public class PrinterTaskRunner extends AbstractTaskRunner{
    public PrinterTaskRunner(TaskRunner taskRunner){
        super(taskRunner);
    }

    @Override
    public void executeOneTask(){
        super.executeOneTask();
        System.out.println("Done: " + LocalDateTime.now().format(Constants.HOUR_FORMATTER));
    }

    @Override
    public void executeAll() {
        while(super.hasTask()) {
            executeOneTask();
        }
    }
}
