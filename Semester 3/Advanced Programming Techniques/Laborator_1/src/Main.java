import expresions.ComplexExpression;
import expresions.ExpressionParser;

public class Main {
    public static void main(String[] args) {
        ExpressionParser parser = new ExpressionParser(args);
        ComplexExpression complexExpression = parser.build();
        System.out.println(complexExpression.execute());
    }
}