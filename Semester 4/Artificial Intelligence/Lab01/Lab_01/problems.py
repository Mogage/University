from domain.customs import Point, MultiDimensionalVector
import heapq

class Laborator_01:
    '''
        Formeaza o lista care contine cuvintele din sir, itereaza prin aceasta si la fiecare pas compara
        daca elementul curent este mai mare(alfabetic) decat cel candidat
    '''
    @staticmethod
    def problem_01(text):
        cuvinte = text.split(' ')

        candidat = cuvinte[0]
        for cuvant in cuvinte[1:]:
            if cuvant > candidat:
                candidat = cuvant

        return candidat

    '''
        Creeaza 2 puncte si face distanta euclidiana intre acestea
    '''
    @staticmethod
    def problem_02(x1, y1, x2, y2):
        firstPoint = Point(x1,y1)
        secondPoint = Point(x2, y2)
        
        return firstPoint.distance(secondPoint)

    ''' 
        Creeaza 2 matrici si le inmulteste, facand produsul scalar
    '''
    @staticmethod
    def problem_03(vector1, vector2):
        firstVector = MultiDimensionalVector(vector1)
        secondVector = MultiDimensionalVector(vector2)
        
        return firstVector.multiply(secondVector)

    '''
        Creeaza 2 set-uri, unul pentru toate cuvintele din sir si altul pentru cuvintele duplicate, formeaza o lista
        care contine cuvintele din sir, itereaza prin aceasta si la fiecare pas daca elementul nu se afla in
        set-ul de cuvinte, este adaugat, altfel il adauga in set-ul de cuvinte duplicate. La final se face diferenta
        intre set-uri
    '''
    @staticmethod
    def problem_04(text):
        cuvinte = set()
        cuvinteDublicat = set()

        cuvinteText = text.split(" ")

        for cuvant in cuvinteText:
            if cuvant in cuvinte:
                cuvinteDublicat.add(cuvant)
            else:
                cuvinte.add(cuvant)

        rezultat = cuvinte.difference(cuvinteDublicat)

        return rezultat

    '''
        Parcurge lista cu numere si face suma acestora. Elementul duplicat se va determina scazand din numarul obtinut
        anterior suma gauss a numerelor pana la n-1.
    '''
    @staticmethod
    def problem_05(elemente):
        nrElemente = len(elemente)
        suma = (nrElemente * (nrElemente - 1)) // 2
        sumaSir = 0
        for element in elemente:
            sumaSir += element
        
        return sumaSir - suma
    
    '''
        Ne vom folosi de un contor si un candidat. Iteram prin vector, iar la fiecare pas daca elementul curent
        este egal cu cel candidat, incrementam contorul, altfel il decrementam, iar cand ajungem la 0 actualizam
        candidatul. La final verificam sa respecte conditia de majoritar
    '''
    @staticmethod
    def problem_06(elemente):
        count = 0
        candidat = None
        for element in elemente:
            if count == 0:
                candidat = element
            if element == candidat:
                count += 1
            else:
                count -= 1

        if elemente.count(candidat) > len(elemente) // 2:
            return candidat
        else:
            return None
    
    '''
        Pentru a determina al k-lea cel mai mare element din lista ne vom folosi de un min-heap in care initial
        adaugam primele k elemente, dupa iteram prin restul listei si de fiecare data cand gasim un element mai
        mare decat cel minim din heap, scoatem minimul si adaugam elementul curent. La final al k-lea cel mai
        mare element va fi cel din varful heap-ului
    '''
    @staticmethod
    def problem_07(elemente, k):
        heap = elemente[:k]
        heapq.heapify(heap)
        for element in elemente[k:]:
            if element > heap[0]:
                heapq.heappop(heap)
                heapq.heappush(heap, element)
        return heap[0]

    '''
        Ne vom folosi de string pentru a reprezenta binar numerele. La fiecare pas cautam cel mai din dreapta 0.
        In cazul in care nu exista, incrementam numarul de cifre din reprezentarea binara. Dupa vom face toate
        elementele 0 aflate in dreapta bit-ului schimbat in 1
    '''
    @staticmethod
    def problem_08(n):
        elemente = []
        binar = '1'
        size = 1

        for _ in range(1, n+1):
            elemente.append(binar)
            pos = binar.rfind('0')
            if pos == -1:
                size += 1
                pos = 0
            binar = binar[:pos] + '1' +binar[pos+1:]
            zeros = ''
            for _ in range(size - pos - 1):
                zeros += '0'
            binar = binar[:pos+1] + zeros

        return elemente
            
    '''
        Verificam cazurile in care punctele nu sunt date in pereche (colt_stg_sus si colt_dr_jos). Ulterior 
        parcurgem sectiunea respectiva si calculam suma. Daca matricea era citita de la utilizator se putea face
        cu sume partiale
    '''
    @staticmethod
    def problem_09(matrix, x1, y1, x2, y2):
        if x1 > x2:
            x1,x2 = x2,x1
        if y1 > y2:
            y1,y2 = y2,y1
        
        sum = 0
        for row in range(x1,x2 + 1):
            for col in range(y1,y2 + 1):
                sum += matrix[row][col]

        return sum
    
    '''
        Cauta pozitia celui mai din dreapta 0 cu o cautare binara si calculeaza numarul de 1-uri din vector
    '''
    @staticmethod
    def _findNumberOfOnes(vector):
        size = len(vector) - 1
        left, right = 0, size
        zero = -1
        while left <= right:
            mid = (left + right) // 2
            if vector[mid] == 0:
                zero = mid
                left = mid + 1
            else:
                right = mid - 1
        
        return size - zero

    '''
        Parcurge linie cu linie matricea si calculeaza numarul de 1-uri dintr-o linie folosind o cautare binara
        pentru a determina cel mai din dreapta 0 si a face diferenta intre pozitia acestuia si lungimea
        liniei. La fiecare pas se actualizeaza, daca este cazul, rezultatul final
    '''
    @staticmethod
    def problem_10(matrix):
        max_ones = 0
        index = -1

        size = len(matrix)
        for row in range(size):
            ones = Laborator_01._findNumberOfOnes(matrix[row])
            if ones > max_ones:
                max_ones = ones
                index = row
        
        return index + 1