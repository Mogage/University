package operations;

import expresions.ComplexExpression;
import utils.Constants;
import complex.ComplexNumber;

public class AdditionExpression extends ComplexExpression {
    public AdditionExpression(Constants.Operation operation, ComplexNumber[] complexNumbers) {
        super(operation, complexNumbers);
    }

    @Override
    public ComplexNumber executeOneOperation(ComplexNumber number1, ComplexNumber number2){
        return number1.addition(number2);
    }
}

