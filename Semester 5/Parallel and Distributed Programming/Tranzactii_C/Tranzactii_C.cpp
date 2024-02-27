#include <fstream>
#include <iostream>
#include <string>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <map>
#include <random>

using namespace std;

const int SIZE = 25;
mutex mtx = mutex();

class Dictionary {
private:
    map<string, map<string, int>> dictionary;
public:
    condition_variable cvReader;
    condition_variable cvWorker;
    atomic<int> counter;
    mutex mtx2;
    Dictionary(): mtx2(mutex()) {
        dictionary = map<string, map<string, int>>();
    }

    bool verifyKey(string key) {
        return dictionary.find(key) != dictionary.end();
    }

    void insertKey(string key) {
        dictionary[key] = map<string, int>();
    }

    void pushValue(string key, string valueKey, int value) {
        mtx2.lock();
        cout << "Adding value \n";
        counter++;
        if (dictionary[key].find(valueKey) == dictionary[key].end()) {
            dictionary[key][valueKey] = value;
        }
        else {
            dictionary[key][valueKey] += value;
        }
        mtx2.unlock();
    }

    void deleteValue(string key, string valueKey) {
        mtx2.lock();
        dictionary[key].erase(valueKey);
        mtx2.unlock();
    }

    map<string, map<string, int>>::iterator iterator() {
        mtx2.lock();
        auto toReturn = dictionary.begin();
        mtx2.unlock();
        return toReturn;
    }

    map<string, map<string, int>>::iterator end() {
        mtx2.lock();
        auto toReturn = dictionary.end();
        mtx2.unlock();
        return toReturn;
    }
};

void worker(Dictionary& dictionary, string client, string operationType) {
    mtx.lock();
    if (!dictionary.verifyKey(client)) {
        dictionary.insertKey(client);
    }
    mtx.unlock();
    for (int i = 0; i < SIZE; i++) {
        dictionary.pushValue(client, operationType, rand() % 5 + 1);
        if (dictionary.counter == 50) {
            dictionary.cvReader.notify_one();
            unique_lock<mutex> lock(mtx);
            dictionary.cvWorker.wait(lock);
            dictionary.counter = 0;
        }
    }
}

void reader(Dictionary& dictionary) {
    while (true) {
        unique_lock<mutex> lock(dictionary.mtx2);
        dictionary.cvReader.wait(lock, [&]() { return dictionary.counter < 50; });
        map<string, map<string, int>>::iterator it = dictionary.iterator();
        while (it != dictionary.end()) {
            cout << it->first << ":\n";
            for (auto it2 = it->second.begin(); it2 != it->second.end(); it2++) {
                cout << "    " + it2->first + " " + to_string(it2->second) + "\n";
            }
            it++;
        }
        dictionary.cvWorker.notify_one();
    }
}


int main()
{
    Dictionary dictionary = Dictionary();

    thread t1(worker, ref(dictionary), "client1", "debit");
    thread t2(worker, ref(dictionary), "client2", "debit");
    thread t3(worker, ref(dictionary), "client1", "credit");
    thread t4(worker, ref(dictionary), "client2", "credit");
    thread t5(reader, ref(dictionary));

    t1.join();
    t2.join();
    t3.join();
    t4.join();
    t5.join();

    return 0;
}

