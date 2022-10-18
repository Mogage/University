package expresions;

import utils.Constants;
import complex.ComplexNumber;

import java.util.Objects;

public class ExpressionParser {
    private final int numberOfArguments;
    private ExpressionFactory expressionFactory;
    private final String[] args;

    public ExpressionParser(String[] args) {
        this.expressionFactory = ExpressionFactory.getInstance();
        this.args = args;
        this.numberOfArguments = args.length;
    }

    private boolean isExpression(String[] args) {
        if (numberOfArguments % 2 == 0) {
            return false;
        }
        for (int count = 1; count < numberOfArguments; count += 2) {
            if (args[count].length() != 1 || !Constants.OPERATORS.contains(args[count]) || !Objects.equals(args[count], args[1])) {
                return false;
            }
        }
        for (int count = 0; count < numberOfArguments; count += 2) {
            if (args[count].length() == 1 && Constants.OPERATORS.contains(args[count])) {
                return false;
            }
        }
        return true;
    }

    private ComplexNumber parseComplexNumber(String ComplexNumberString) {
        boolean realHasSign = false, imaginaryHasSign = false;
        double real, imaginary;
        int operatorIndex;

        if (Constants.OPERATORS.contains(String.valueOf(ComplexNumberString.charAt(0)))) {
            realHasSign = true;
            ComplexNumberString = ComplexNumberString.substring(1);
        }

        if (ComplexNumberString.indexOf('+') != -1) {
            operatorIndex = ComplexNumberString.indexOf('+');
        } else {
            imaginaryHasSign = true;
            operatorIndex = ComplexNumberString.indexOf('-');
        }

        if (operatorIndex == -1) {
            real = Integer.parseInt(ComplexNumberString);
            imaginary = 0;
        } else {
            real = Integer.parseInt(ComplexNumberString.substring(0, operatorIndex));
            imaginary = Integer.parseInt(ComplexNumberString.substring(operatorIndex + 1, ComplexNumberString.length() - 2));
        }

        if (realHasSign) {
            real = real * -1;
        }
        if (imaginaryHasSign) {
            imaginary = imaginary * -1;
        }

        return new ComplexNumber(real, imaginary);
    }

    private ComplexNumber[] getComplexNumbers() {
        int complexNumbersLength = numberOfArguments / 2 + 1;
        ComplexNumber[] complexNumbers = new ComplexNumber[complexNumbersLength];
        for (int count = 0; count < numberOfArguments; count += 2) {
            complexNumbers[count / 2] = parseComplexNumber(args[count]);
            //System.out.println(complexNumbers[count / 2]);
        }
        return complexNumbers;
    }

    public ComplexExpression build() {
        if (!isExpression(args)) {
            System.out.println("Params don't form an expression.");
            System.exit(1);
        }
        ComplexNumber[] complexNumbers = getComplexNumbers();
        return expressionFactory.createExpression(Constants.operatorsMap.get(args[1]), complexNumbers);
    }
}
