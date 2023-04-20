from math import sqrt

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    @staticmethod
    def _calculateXEuclidian(xPoint1, xPoint2):
        return (xPoint1 - xPoint2)*(xPoint1 - xPoint2)
    
    @staticmethod
    def _calculateYEuclidian(yPoint1, yPoint2):
        return (yPoint1 - yPoint2)*(yPoint1- yPoint2)

    def distance(self, point2):
        return sqrt(self._calculateXEuclidian(self.x, point2.x) + self._calculateYEuclidian(self.y, point2.y))

class MultiDimensionalVector:
    def __init__(self, elemente):
        self.nrDimensiuni = len(elemente)
        self.nrElemente = len(elemente[0])
        self.elemente = elemente

    def multiply(self, anotherVector):
        produsScalar = 0
        for linie in range(0, self.nrDimensiuni):
            for coloana in range(0, self.nrElemente):
                produsScalar += self.elemente[linie][coloana] * anotherVector.elemente[linie][coloana]
        return produsScalar
