//////////////// Program 1 ////////////////

#include <iostream>
using namespace std;
const double PI = 3.145;

double Perimetru(double raza) {
    return 2*PI*raza;
}

double Arie(double raza) {
    return PI*raza*raza;
}

int main() {
    double raza;
    cin >> raza;
    cout << Perimetru(raza) << Arie(raza);
    return 0;
}

//////////////// Program 2 ////////////////

#include <iostream>
using namespace std;

int cmmdc(int a, int b){
    int r = a % b;
    while(r) {
        a = b;
        b = r;
        r = a % b;
    }
    return b;
}

int main(){
    int a, b;
    cin >> a >> b;
    cout << cmmdc(a,b);
    return 0;
}

//////////////// Program 3 ////////////////

#include <iostream>
using namespace std;

int main() {
    int n, sum=0, x;
    cin >> n;
    for (int i = 0; i < n; i++) {
        cin >> x;
        sum = sum + x;
    }
    cout << sum;
    return 0;
}