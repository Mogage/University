#include <string>
#include <fstream>
#include <iostream>
#include <math.h>
using namespace std;
ifstream fin("in.txt");
ofstream fout("out.txt");

int main(){
    int realNumber;
    fin >> realNumber;
    fout << sqrt(realNumber);
    return 0;
}