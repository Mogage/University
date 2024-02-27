%{
    #include <iostream>
    #include <fstream>
    #include <string>
    #include <vector>
    #include <queue>
    #include <stack>
    #include <set>
    #include <math.h>
    #include <string.h>
    extern int yylex();
    extern int yyparse();
    extern FILE* yyin;
    extern int yylineno;

    void yyerror(const char* s);
    void addDeclaration(std::string s,  std::string sValue = "0", std::string type = " dd ");
    void printImports();
    void printDeclarations();
    void printCode();

    std::vector<std::string> imports;
    std::vector<std::string> code;
    std::vector<std::string> tokens;
    std::set<std::string> declarations;
    std::string outputFileName = "outputCode.asm";
    std::ofstream outputFile;

    bool isOperator(std::string c);
    bool isNumber(const std::string& s);
    void evaluateRPN();
%}

%union {
	char* value;
}

%token LID
%token ID 
%token CONST 
%token INCLUDE 
%token USING 
%token NAMESPACE 
%token STD
%token IOSTREAM MAIN
%token INT 
%token CIN COUT RETURN
%token LEFT_BRACE RIGHT_BRACE LEFT_PAREN RIGHT_PAREN
%token COMMA SEMICOLON 
%token LESS_THAN GREATER_THAN EQUAL
%token PLUS MINUS MULTIPLY DIVIDE
%token LEFT_SHIFT RIGHT_SHIFT HASH INCREMENT DECREMENT

%start program

%%

program:
    HASH INCLUDE LESS_THAN IOSTREAM GREATER_THAN
    USING NAMESPACE STD SEMICOLON 
    INT MAIN LEFT_PAREN RIGHT_PAREN
    corp
    {
        imports.push_back("scanf");
        imports.push_back("printf");
        imports.push_back("exit");
        addDeclaration("format", "\"%d\", 0", " db ");
    }
;

corp:
    LEFT_BRACE instr_corp RIGHT_BRACE
;

instr_corp:
    instr instr_corp
    | instr
;

instr:
    lista_decl_instr
    | atribuire SEMICOLON
    | instr_return
    | instr_intrare
    | instr_iesire
;

lista_decl_instr:
    INT decl_instr SEMICOLON
;

decl_instr:
    ID COMMA decl_instr { addDeclaration($<value>1); }
    | ID atribuire_valoare COMMA decl_instr { addDeclaration($<value>1, $<value>2); code.push_back("pop dword [" + std::string($<value>1) + "]"); }
    | ID { addDeclaration($<value>1); }
    | ID atribuire_valoare  { addDeclaration($<value>1, $<value>2); code.push_back("pop dword [" + std::string($<value>1) + "]"); }
;

atribuire:
    ID atribuire_valoare 
    {
        code.push_back("pop dword [" + std::string($<value>1) + "]\n"); 
    }
;

atribuire_valoare:
    EQUAL expr_aritmetica 
    { 
        $<value>$ = strdup($<value>2);
        evaluateRPN();
        tokens.clear();
    }
;

expr_aritmetica:
    term PLUS expr_aritmetica { tokens.push_back("+"); }
    | term MINUS expr_aritmetica { tokens.push_back("-"); }
    | term
;

term :
    factor MULTIPLY term { tokens.push_back("*"); }
    | factor DIVIDE term { tokens.push_back("/"); }
    | factor 
;

factor:
    ID { tokens.push_back("[" + std::string($<value>1) + "]"); }
    | CONST { tokens.push_back($<value>1); }
;

instr_intrare:
    CIN  RIGHT_SHIFT ID SEMICOLON { code.push_back("push dword " + std::string($<value>3) + "\n\t\tpush dword format\n\t\tcall [scanf]\n\t\tadd ESP, 4 * 2\n"); }
;

instr_iesire:
    COUT LEFT_SHIFT ID SEMICOLON { code.push_back("push dword [" + std::string($<value>3) + "]\n\t\tpush dword format\n\t\tcall [printf]\n\t\tadd ESP, 4 * 2\n"); }
;

instr_return:
    RETURN CONST SEMICOLON
;

%%

void yyerror(const char* s)
{
    std::cout << "Syntax error at line: " << yylineno << '\n';
    exit(1);
}

int main(){
    FILE *fp;
    char filename[50];
    std::cout << "Enter file name: ";
    scanf("%s", filename);
    fp = fopen(filename, "r");
    yyin = fp;
    yyparse();
    std::cout << "The program is syntax corect";

    outputFile.open(outputFileName);
    outputFile << "bits 32\nglobal start\n\n";

    printImports();
    outputFile << "segment data use32 class=data\n";
    printDeclarations();

    outputFile << "\nsegment code use32 class=code\n\tstart:\n";
    printCode();
    outputFile << "\t\tpush dword 0\n\t\tcall [exit]";

    return 0;
}

void addDeclaration(std::string s, std::string sValue, std::string type)
{
    if (declarations.find(s) != declarations.end())
    {
        return;
    }

    declarations.insert(s + type + sValue);
}

void printImports()
{
    for (auto import : imports)
    {
        outputFile << "extern " << import << "\nimport " << import << " msvcrt.dll\n\n";
    }
}

void printDeclarations()
{
    for (auto declaration : declarations)
    {
        outputFile << "\t" + declaration + "\n";
    }
}

void printCode()
{
    for (auto codeLine : code)
    {
        outputFile << "\t\t" + codeLine + "\n";
    }
}

bool isOperator(std::string c) {
    return c == "+" || c == "-" || c == "*" || c == "/";
}

bool isNumber(const std::string& s)
{
    std::string::const_iterator it = s.begin();
    while (it != s.end() && std::isdigit(*it)) ++it;
    return !s.empty() && it == s.end();
}

void evaluateRPN() {
    std::stack<double> valueStack;

    for (auto token : tokens) {
        if (isOperator(token)) {
            double b = valueStack.top();
            valueStack.pop();
            double a = valueStack.top();
            valueStack.pop();

            if (token == "+") {
                valueStack.push(a + b);
                code.push_back("pop EDX");
                code.push_back("pop EAX");
                code.push_back("add AX, DX");
                code.push_back("push EAX\n");
            } else if (token == "-") {
                valueStack.push(a - b);
                code.push_back("pop EDX");
                code.push_back("pop EAX");
                code.push_back("sub AX, DX");
                code.push_back("push EAX\n");
            } else if (token == "*") {
                valueStack.push(a * b);
                code.push_back("pop EDX");
                code.push_back("pop EAX");
                code.push_back("mul DL");
                code.push_back("push EAX\n");
            } else if (token == "/") {
                valueStack.push(a / b);
                code.push_back("pop EBX");
                code.push_back("pop EAX");
                code.push_back("mov DX, 0");
                code.push_back("div WORD BX");
                code.push_back("push EAX\n");
            }
        } else if (isNumber(token)) {
            valueStack.push(std::stod(token));
            code.push_back("push " + token);
        } else {
            code.push_back("push dword " + token);
            valueStack.push(5);
        }
    }
}