%option noyywrap
%option yylineno

%{ 				
#include "Parser.h"
#include <iostream>
#include <fstream>
#include <string.h>
#include <vector>
std::ofstream fout("result.txt");

bool areTreeCreated = false;
int TSCode;
struct Node {
    std::string data;
    int index;
    Node* left;
    Node* right;

    Node(std::string value) : data(value), left(nullptr), right(nullptr) {}
};

Node* insert(Node* root, std::string value) {
    if (root == nullptr) {
        return new Node(value);
    }

    if (value < root->data) {
        root->left = insert(root->left, value);
    }
    else {
        root->right = insert(root->right, value);
    }

    return root;
}

Node* search(Node* root, std::string value) {
    if (root == nullptr) {
        return nullptr;
    }

    if (value == root->data) {
        return root;
    }
    else if (value < root->data) {
        return search(root->left, value);
    }
    else {
        return search(root->right, value);
    }
}

void inOrderTraversal(Node* root) {
    if (root == nullptr) {
        return;
    }

    inOrderTraversal(root->left);
    fout << root->data << " " << TSCode << "\n";
    root->index = TSCode;
    TSCode++;
    inOrderTraversal(root->right);
}

    Node* identifiersTree = nullptr;
    Node* constantsTree = nullptr;

void handleContent(char* yytext, int code)
{
    if (!areTreeCreated)
    {
        return;
    }
    // fout << yytext << " -> " << code << '\n';
    fout << code << " -\n";
}
%} 

/* Definitions */
LID     [a-zA-Z_][a-zA-Z0-9_]*
LCONST  [-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?
WS      [ \t\r\n]

%%

"#"             { handleContent(yytext, HASH); }
"<<"            { handleContent(yytext, LEFT_SHIFT); }
">>"            { handleContent(yytext, RIGHT_SHIFT); }
"=="            { handleContent(yytext, EQUAL); }
"!="            { handleContent(yytext, NOT_EQUAL); }
"<"             { handleContent(yytext, LESS_THAN); }
">"             { handleContent(yytext, GREATER_THAN); }
"<="            { handleContent(yytext, LESS_THAN_OR_EQUAL); }
">="            { handleContent(yytext, GREATER_THAN_OR_EQUAL); }
"{"             { handleContent(yytext, LEFT_BRACE); }
"}"             { handleContent(yytext, RIGHT_BRACE); }
"("             { handleContent(yytext, LEFT_PAREN); }
")"             { handleContent(yytext, RIGHT_PAREN); }
";"             { handleContent(yytext, SEMICOLON); }
","             { handleContent(yytext, COMMA); }
"+"             { handleContent(yytext, PLUS); }
"-"             { handleContent(yytext, MINUS); }
"*"             { handleContent(yytext, MULTIPLY); }
"/"             { handleContent(yytext, DIVIDE); }
"%"             { handleContent(yytext, MODULO); }
"="             { handleContent(yytext, EQUAL); }
"++"            { handleContent(yytext, INCREMENT); }
"--"            { handleContent(yytext, DECREMENT); }
"cout"          { handleContent(yytext, COUT); }
"cin"           { handleContent(yytext, CIN); }
"return"        { handleContent(yytext, RETURN); }
"using"         { handleContent(yytext, USING); }
"namespace"     { handleContent(yytext, NAMESPACE); }
"std"           { handleContent(yytext, STD); }
"include"       { handleContent(yytext, INCLUDE); }
"iostream"      { handleContent(yytext, IOSTREAM); }
"math.h"        { handleContent(yytext, MATH_H); }
"string.h"      { handleContent(yytext, STRING_H); }
 
"const"         { handleContent(yytext, CONST_); }
"int"           { handleContent(yytext, INT); }
"float"         { handleContent(yytext, FLOAT); }
"double"        { handleContent(yytext, DOUBLE); }
"void"          { handleContent(yytext, VOID); }
"struct"        { handleContent(yytext, STRUCT); }
"const"         { handleContent(yytext, CONST); }
"if"            { handleContent(yytext, IF); }
"while"         { handleContent(yytext, WHILE); }
"for"           { handleContent(yytext, FOR); }
 
{LCONST}        { 
                    if (areTreeCreated) {
                        // fout << yytext << " -> " << CONST << " " << search(constantsTree, yytext)->index << '\n'; 
                        fout << CONST << " " << search(constantsTree, yytext)->index << '\n'; 
                    }
                    
                    if (!search(constantsTree, yytext)) {
                        constantsTree = insert(constantsTree, yytext);
                    }
                }
{LID}           { 
                    if (areTreeCreated) {
                        // yyleng <= 8 ? fout << yytext << " -> " << ID  << " " << search(identifiersTree, yytext)->index << '\n' : 
                        yyleng <= 8 ? fout << ID  << " " << search(identifiersTree, yytext)->index << '\n' : 
                                        std::cout << "Error at line " << yylineno << ". Text too long: " << yytext << '\n';     
                    } else if (yyleng > 8) {
                        std::cout << "Error at line " << yylineno << ". Text too long: " << yytext << '\n'; 
                        exit(-1);
                    }
                    if (!search(identifiersTree, yytext)) {
                        identifiersTree = insert(identifiersTree, yytext);
                    }
                }
{WS}+           ; 
.               { std::cout << "Error at line " << yylineno << ". Unexpected character: " << yytext << '\n'; }

%%

int main(){
    FILE *fp;
    char path[100] = "inputFiles\\";
    char filename[50];
    std::cout << "Enter file name: ";
    scanf("%s", filename);
    strcat(path, filename);
    fp = fopen(path, "r");
    yyin = fp;
    yylex();
    
    TSCode = 0;
    fout << "Identifiers Tree: \n";
    inOrderTraversal(identifiersTree);
    TSCode = 0;
    fout << "\nConstants Tree: \n";
    inOrderTraversal(constantsTree);
    fout << '\n';

    areTreeCreated = true;

    fp = fopen(path, "r");
    yyin = fp;
    yylex();

    return 0;
}