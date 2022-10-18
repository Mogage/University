package complex;

import utils.Constants;

public class ComplexNumber {
    private double realPart, imaginaryPart;

    public double getRealPart() {
        return realPart;
    }

    public double getImaginaryPart() {
        return imaginaryPart;
    }

    public ComplexNumber(double real, double imaginary) {
        realPart = real;
        imaginaryPart = imaginary;
    }

    public ComplexNumber addition (ComplexNumber SecondNumber) {
        this.realPart = this.realPart + SecondNumber.realPart;
        this.imaginaryPart = this.imaginaryPart + SecondNumber.imaginaryPart;
        return this;
    }

    public ComplexNumber subtraction(ComplexNumber SecondNumber) {
        this.realPart = this.realPart - SecondNumber.realPart;
        this.imaginaryPart = this.imaginaryPart - SecondNumber.imaginaryPart;
        return this;
    }

    public ComplexNumber multiplication (ComplexNumber SecondNumber) {
        this.realPart = this.realPart * SecondNumber.realPart - this.imaginaryPart * SecondNumber.imaginaryPart;
        this.imaginaryPart = this.realPart * SecondNumber.imaginaryPart + this.imaginaryPart * SecondNumber.realPart;
        return this;
    }

    public ComplexNumber division (ComplexNumber SecondNumber) {
        double denominator = SecondNumber.realPart * SecondNumber.realPart + SecondNumber.imaginaryPart * SecondNumber.imaginaryPart;
        double realNumerator = this.realPart * SecondNumber.realPart + this.imaginaryPart * SecondNumber.imaginaryPart;
        double imaginaryNumerator = this.imaginaryPart * SecondNumber.realPart - this.realPart * SecondNumber.imaginaryPart;
        this.realPart = realNumerator / denominator;
        this.imaginaryPart = imaginaryNumerator / denominator;
        return this;
    }

    public String toString() {
        return "Real: " + realPart + "\tImaginary: " + imaginaryPart;
    }
}

