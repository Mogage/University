#include < iostream >
using namespace std ;

int main ( ) {
    int n , sum = 0 , x ;
    cin >> n ;
    for ( int i = 0 ; i < n ; i = i + 1 ) {
        cin >> x ;
        sum = sum + x ;
    }
    cout << sum ;
    return 0 ;
}