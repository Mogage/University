#include <iostream>
using namespace std;

int cmmdc (int a, int b) {
    int r = a % b;
    while (r) {
        a = b;
        b = r;
        r = a % b;
    }
    return b;
}

int main () {
    int a, b;
    cin >> a >> b;
    cout << cmmdc(a, b);
    return 0;
}