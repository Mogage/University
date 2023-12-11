%{
    #include <iostream>
    #include <string>
    #include <string.h>
    extern int yylex();
    extern int yyparse();
    extern FILE* yyin;
    extern int yylineno;
    void yyerror(const char* s);
%}



%token LID
%token ID 
%token CONST 
%token INCLUDE 
%token USING 
%token NAMESPACE 
%token STD
%token IOSTREAM MATH_H STRING_H 
%token CONST_ INT FLOAT DOUBLE VOID STRUCT
%token IF WHILE FOR CIN COUT RETURN
%token LEFT_BRACE RIGHT_BRACE LEFT_PAREN RIGHT_PAREN
%token COMMA SEMICOLON 
%token LESS_THAN GREATER_THAN EQUAL NOT_EQUAL EQUAL_EQUAL LESS_THAN_OR_EQUAL GREATER_THAN_OR_EQUAL
%token PLUS MINUS MULTIPLY DIVIDE MODULO
%token LEFT_SHIFT RIGHT_SHIFT HASH INCREMENT DECREMENT

%start program

%%

program:
    HASH INCLUDE LESS_THAN tip_include GREATER_THAN
    USING NAMESPACE STD SEMICOLON 
    decl_globale
;

tip_include:
    IOSTREAM 
    | MATH_H
    | STRING_H
;

decl_globale:
    tip_decl_globale decl_globale
    | tip_decl_globale
;

tip_decl_globale:
    constanta
    | functie
;

constanta:
    CONST_ tip atribuire SEMICOLON
;

functie:
    tip ID LEFT_PAREN list_decl RIGHT_PAREN
    corp 
;

tip:
    INT
    | FLOAT
    | DOUBLE
    | VOID
    | tip_struct
;

tip_struct:
    STRUCT ID LEFT_BRACE list_decl_struct RIGHT_BRACE
;

list_decl_struct:
    list_decl SEMICOLON list_decl_struct
    | list_decl SEMICOLON
;

list_decl:
    decl COMMA list_decl
    | decl
    | ;
;

decl:
    tip atribuire
    | tip ID
;

corp:
    LEFT_BRACE instr_corp RIGHT_BRACE
    | instr_return
;

instr_corp:
    instr instr_corp
    | instr
;

instr:
    lista_decl_instr
    | atribuire SEMICOLON
    | instr_if
    | instr_cicl
    | instr_return
    | instr_intrare
    | instr_iesire
    | apel_functie
;

lista_decl_instr:
    tip decl_instr SEMICOLON
    | tip_struct SEMICOLON
;

decl_instr:
    atribuire
    | ID
;

apel_functie:
    ID LEFT_PAREN list_param RIGHT_PAREN
;

list_param:
    expr_aritmetica COMMA list_param
    | expr_aritmetica
;

instr_if:
    IF LEFT_PAREN cond RIGHT_PAREN corp
;

atribuire:
    ID EQUAL expr_aritmetica
    | ID EQUAL apel_functie
;

expr_aritmetica:
    expr_aritmetica op_aritmetica expr_aritmetica
    | ID INCREMENT
    | ID DECREMENT
    | ID
    | CONST
;

instr_intrare:
    CIN list_intrari SEMICOLON
;

list_intrari:
    RIGHT_SHIFT ID list_intrari
    | RIGHT_SHIFT ID
;

instr_cicl:
    WHILE LEFT_PAREN cond RIGHT_PAREN corp
    | FOR LEFT_PAREN decl SEMICOLON cond SEMICOLON atribuire RIGHT_PAREN corp
;

cond:
    ID
    | CONST
    | expr_aritmetica op_rel expr_aritmetica
;

instr_iesire:
    COUT list_iesiri SEMICOLON
;

list_iesiri:
    LEFT_SHIFT tip_iesire list_iesiri
    | LEFT_SHIFT tip_iesire
;

tip_iesire:
    expr_aritmetica
    | apel_functie
;

op_rel:
    NOT_EQUAL
    | EQUAL_EQUAL
    | GREATER_THAN
    | LESS_THAN
    | GREATER_THAN_OR_EQUAL
    | LESS_THAN_OR_EQUAL
;

op_aritmetica:
    PLUS
    | MULTIPLY
    | DIVIDE
    | MODULO
    | MINUS
;

instr_return:
    RETURN expr_aritmetica SEMICOLON
;

%%

void yyerror(const char* s)
{
    std::cout << "Syntax error at line: " << yylineno << '\n';
    exit(1);
}

int main(){
    FILE *fp;
    char path[100] = "inputFiles\\";
    char filename[50];
    std::cout << "Enter file name: ";
    scanf("%s", filename);
    strcat(path, filename);
    fp = fopen(path, "r");
    yyin = fp;
    yyparse();
    std::cout << "The program is syntax corect";
    return 0;
}