//////////////// Program 1 ////////////////
#include <math>
using namespace std;

int main(){
    int number;
    fin >> number;
    fout << sqrt(number);
    return 0;
}

//////////////// Program 2 ////////////////

#include <string>
#include <fstream>
#include <iostream>
#include <math.h>
using namespace std;
ifstream fin("in.txt");
ofstream fout("out.txt");

int main(){
    int number;
    fin >> number;
    fout << sqrt(number);
    return 0;
}