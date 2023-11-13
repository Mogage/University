#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <map>
#include <algorithm>

std::string fileName = "Exercise_2_1.cpp";
const std::string inPath = "Part_1\\" + fileName;
const std::string outPath = "Out_" + fileName.replace(fileName.length() - 4, 4, ".txt");
std::ifstream in(inPath);
std::ofstream out(outPath);

std::map<std::string, int> idTable;

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
    std::cout << root->data << " " << TSCode << "\n";
    root->index = TSCode;
    TSCode++;
    inOrderTraversal(root->right);
}


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

bool IsNumber(const std::string& str) {

    try {
        size_t pos;
        if (str.find('.') != std::string::npos) {
            double d = std::stod(str, &pos);
            if (pos == str.length()) {
                return true;
            }
        }
        else {
            int i = std::stoi(str, &pos);
            if (pos == str.length()) {
                return true;
            }
        }
    }
    catch (std::invalid_argument& e) {
    }
    catch (std::out_of_range& e) {
    }

    return false;
}

void TransformCode(std::ostream& outStream) {
    std::string line;
    int lineNumber = 0;
    Node* identifiersTree = nullptr;
    Node* constantsTree = nullptr;
    std::vector<std::string> words;

    while (std::getline(in, line)) {
        std::stringstream lineStream(line);
        std::string word;
        lineNumber++;
        while (std::getline(lineStream, word, ' ')) {
            words.push_back(word);
            switch (idTable[word]) {
            case 0: {
                if (word.length() >= 8) {
                    outStream << "Error at line " << lineNumber << ": " << word << " is not a valid identifier\n";
                    break;
                }
                if (IsNumber(word)) {
                    if (!search(constantsTree, word)) {
                        constantsTree = insert(constantsTree, word);
                        //outStream << word << " 1\n";
                    }
                    break;
                }
                else if (!word.empty()) {
                    if (!search(identifiersTree, word)) {
                        identifiersTree = insert(identifiersTree, word);
                        //outStream << word << " 0\n";
                    }
                    break;
                }
                break;
            }
            default: {
                //outStream << idTable[word] << " -\n";
                break;
            }
            }
        }
    }

    outStream << "Identifiers:\n";
    TSCode = 0;
    inOrderTraversal(identifiersTree);
    outStream << "\nConstants:\n";
    TSCode = 0;
    inOrderTraversal(constantsTree);
    outStream << "\n\n";


    for (const auto word : words) {
        switch (idTable[word]) {
        case 0: {
            if (IsNumber(word)) {
                outStream << "1 " << search(constantsTree, word)->index << "\n";
            }
            else if (!word.empty()) {
                outStream << "0 " << search(identifiersTree, word)->index << "\n";
            }
            break;
        }
        default: {
            outStream << idTable[word] << " -\n";
            break;
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
