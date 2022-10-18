package expresions;

import operations.AdditionExpression;
import operations.DivisionExpression;
import operations.MultiplicationExpression;
import operations.SubtractionExpression;
import utils.Constants.Operation;
import complex.ComplexNumber;

public class ExpressionFactory {
    private static final ExpressionFactory instance = new ExpressionFactory();

    private ExpressionFactory() {}

    public static ExpressionFactory getInstance() {
        return instance;
    }

    public ComplexExpression createExpression(Operation operation, ComplexNumber[] args) {
        switch (operation) {
            case ADDITION -> {
                return new AdditionExpression(operation, args);
            }
            case SUBTRACTION -> {
                return new SubtractionExpression(operation, args);
            }
            case MULTIPLICATION -> {
                return new MultiplicationExpression(operation, args);
            }
            case DIVISION -> {
                return new DivisionExpression(operation, args);
            }
            default -> {
                return null;
            }
        }
    }
}
