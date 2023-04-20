package root.model;

import java.io.Serializable;
import java.time.format.DateTimeFormatter;

public class Constants implements Serializable {
    public static final DateTimeFormatter DATE_FORMATTER = DateTimeFormatter.ofPattern("dd/MM/yyyy");
    public static final DateTimeFormatter TIME_FORMATTER = DateTimeFormatter.ofPattern("HH:mm");
}
