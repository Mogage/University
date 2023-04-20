from random import randint
from utils import generateNewValue

class Chromosone():
    def __init__(self, noOfNodes, com) -> None:
        self.__representation = [randint(1, 1000) for _ in range(noOfNodes)]
        self.__fitness = 0.0

    @property
    def representation(self):
        return self.__representation
    
    @property
    def fitness(self):
        return self.__fitness
    
    @representation.setter
    def representation(self, newRep = []):
        self.__representation = newRep

    @fitness.setter
    def fitness(self, newFitness = 0.0):
        self.__fitness = newFitness

    def crossover(self, c):
        size = len(self.__representation)
        r = randint(0, size - 1)
        newrepres = []
        for i in range(r):
            newrepres.append(self.__representation[i])
        for i in range(r, size):
            newrepres.append(c.__representation[i])
        offspring = Chromosone(size, 2)
        offspring.__representation = newrepres
        return offspring

    def mutation(self):
        index1 = randint(0, len(self.__representation) - 1)
        index2 = randint(0, len(self.__representation) - 1)
        while index1 == index2:
            index2 = randint(0, len(self.__representation) - 1)
        #self.__representation[index1:index2].reverse()
        self.__representation[index1], self.__representation[index2] = self.__representation[index2], self.__representation[index1]

    def __gt__(self, other):
        return self.fitness > other.fitness
    
    def __lt__(self, other):
        return self.fitness < other.fitness
    
    def __eq__(self, other):
        return self.fitness == other.fitness
    
    def __iter__(self):
        self.__position = 0
        return self

    def __next__(self):
        if self.__position >= len(self.__representation):
            raise StopIteration

        toReturn = self.__representation[self.__position]
        self.__position += 1
        return toReturn

    def __len__(self):
        return len(self.__representation)
    
    def __str__(self):
        return str(self.__representation)