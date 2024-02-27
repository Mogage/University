import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;

public class IOHandler {
    public synchronized static void emptyFile() {
        try (BufferedWriter fileWriter = new BufferedWriter(new FileWriter("tranzactii.log", false))){
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
            //System.out.print(text);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }
}
