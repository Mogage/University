#include <iostream>
using namespace std;
const double PI = 3.145;

double Perim (double raza) {
    return 2 * PI * raza;
}

double Arie (double raza) {
    return PI * raza * raza;
}

int main () {
    double raza;
    cin >> raza;
    cout << Perim(raza) << Arie(raza);
    return 0;
}