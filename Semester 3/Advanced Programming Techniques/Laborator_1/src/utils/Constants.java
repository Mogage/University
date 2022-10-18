package utils;

import expresions.ComplexExpression;
import operations.AdditionExpression;

import java.util.HashMap;
import java.util.Map;

public class Constants {
    public static final String OPERATORS = "+-*/";

    public enum Operation {
        ADDITION,
        SUBTRACTION,
        MULTIPLICATION,
        DIVISION
    }

    public static Map<String, Operation> operatorsMap = new HashMap<>() {{
        put("+", Operation.ADDITION);
        put("-", Operation.SUBTRACTION);
        put("*", Operation.MULTIPLICATION);
        put("/", Operation.DIVISION);
    }};
}
