#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <map>
#include <algorithm>

std::string fileName = "Exercise_2_2.cpp";
const std::string inPath = "D:\\1. Repositories\\University\\Semester 5\\Formal Languages and Compilation Techniques\\Lab_1\\" + fileName;
const std::string outPath = "Out_" + fileName.replace(fileName.length() - 4, 4, ".txt");
std::ifstream in(inPath);
std::ofstream out(outPath);

std::map<std::string, int> idTable;

void CreateIdTable() {
    std::vector<std::string> tableContents = {
        "ID", "CONST", "#include", "using", "namespace", "std", "iostream", "math.h", "string.h",
        "const", "int", "float", "double", "void", "struct",
        "if", "while", "for", "cin", "cout", "return",
        "{", "}", "(", ")", ",",  "<" , ">", ";", "=", "!=", "==", "<=", ">=", "+", "-", "*", "/", "%", "<<", ">>"
    };
    int id = 0;

    for (const auto content : tableContents) {
        idTable[content] = id++;
    }
}

bool IsNumber(const std::string& s) {
    return !s.empty() && std::all_of(s.begin(), s.end(), ::isdigit);
}

void TransformCode(std::ostream& outStream) {
    std::string line;

    while (std::getline(in, line)) {
        std::stringstream lineStream(line);
        std::string word;

        while (std::getline(lineStream, word, ' ')) {
            switch (idTable[word]) {
            case 0: {
                if (IsNumber(word)) {
                    outStream << word << " 1\n";
                }
                else if (!word.empty()) {
                    outStream << word << " 0\n";
                }
                break;
            }
            default: {
                outStream << word << " " << idTable[word] << "\n";
                break;
            }
            }
        }
    }
}

int main()
{
    CreateIdTable();
    TransformCode(out);
    return 0;
}