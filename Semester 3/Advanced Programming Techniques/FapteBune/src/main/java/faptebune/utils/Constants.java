package faptebune.utils;

import java.time.format.DateTimeFormatter;

public class Constants {
    public enum ORASE { Cluj_Napoca, Iasi, Bucuresti, Suceava }

    public static final DateTimeFormatter DATE_TIME_FORMATTER = DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm");
}
