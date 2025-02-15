import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;

public class IOHandler {
    public synchronized static void cleanFile() {
        try (BufferedWriter fileWriter = new BufferedWriter(new FileWriter("tranzactii.log"))) {
            fileWriter.write("");
            fileWriter.flush();
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    public synchronized static void writeText(String text) {
        try (BufferedWriter fileWriter = new BufferedWriter(new FileWriter("tranzactii.log", true))){
            fileWriter.write(text);
            fileWriter.flush();
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }
}
