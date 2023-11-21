#ifndef __PARSER_H__
#define __PARSER_H__

/*

"ID", "CONST", "include", "using", "namespace", "std", "iostream", "math.h", "string.h",
            "const", "int", "float", "double", "void", "struct",
            "if", "while", "for", "cin", "cout", "return",
            "{", "}", "(", ")", ",", "<", ">", ";", "=", "!=", "==", "<=", ">=", "+", "-", "*", "/", "%", "<<", ">>",
            "#", "++", "--"
            
            define this list in following index order
*/

#define ID 0
#define CONST 1
#define INCLUDE 2
#define USING 3
#define NAMESPACE 4
#define STD 5
#define IOSTREAM 6
#define MATH_H 7
#define STRING_H 8
#define CONST_ 9
#define INT 10
#define FLOAT 11
#define DOUBLE 12
#define VOID 13
#define STRUCT 14
#define IF 15
#define WHILE 16
#define FOR 17
#define CIN 18
#define COUT 19
#define RETURN 20
#define LEFT_BRACE 21
#define RIGHT_BRACE 22
#define LEFT_PAREN 23
#define RIGHT_PAREN 24
#define COMMA 25
#define LESS_THAN 26
#define GREATER_THAN 27
#define SEMICOLON 28
#define EQUAL 29
#define NOT_EQUAL 30
#define EQUAL_EQUAL 31
#define LESS_THAN_OR_EQUAL 32
#define GREATER_THAN_OR_EQUAL 33
#define PLUS 34
#define MINUS 35
#define MULTIPLY 36
#define DIVIDE 37
#define MODULO 38
#define LEFT_SHIFT 39
#define RIGHT_SHIFT 40
#define HASH 41
#define INCREMENT 42
#define DECREMENT 43

#endif // __PARSER_H__