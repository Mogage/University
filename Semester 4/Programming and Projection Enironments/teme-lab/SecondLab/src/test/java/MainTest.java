import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.DisplayName;

import static org.junit.jupiter.api.Assertions.assertEquals;
import model.ComputerRepairRequest;

public class MainTest {
    @Test
    @DisplayName("First Test")
    public void test1() {
        ComputerRepairRequest crr = new ComputerRepairRequest();
        assertEquals("", crr.getOwnerName());
        assertEquals("", crr.getOwnerAddress());
    }

    @Test
    @DisplayName("Second test")
    public void test2() {
        assertEquals(2, 2, "egale");
    }
}
