#include <iostream>
using namespace std;

int main () {
    int n;
    int sum = 0;
    int x;
    cin >> n;
    for (int i = 0; i < n; i = i++) {
        cin >> x;
        sum = sum + x;
    }
    cout << sum;
    return 0;
}