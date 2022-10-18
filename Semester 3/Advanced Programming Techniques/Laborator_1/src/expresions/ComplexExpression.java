package expresions;

import utils.Constants.Operation;
import complex.ComplexNumber;

public abstract class ComplexExpression {
    private final Operation operation;
    private final ComplexNumber[] complexNumbers;

    public ComplexExpression(Operation operation, ComplexNumber[] complexNumbers) {
        this.operation = operation;
        this.complexNumbers = complexNumbers;
    }

    public abstract ComplexNumber executeOneOperation(ComplexNumber number1, ComplexNumber number2);

    public ComplexNumber execute() {
        ComplexNumber result = new ComplexNumber(complexNumbers[0].getRealPart(), complexNumbers[0].getImaginaryPart());
        for(int count = 1; count < complexNumbers.length; count ++){
            result = executeOneOperation(result, complexNumbers[count]);
        }
        return result;
    }
}

