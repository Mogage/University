from random import randint
from numpy import random


class ShortestPath:
    def __init__(self, graph):
        self.__fitness = 0
        self.__length = 0
        self.__representation = []
        self.__graph = graph
        self.__initRepresentation()

    def __initRepresentation(self):
        self.__length = self.__graph.number_of_nodes()
        self.__representation = [randint(0, self.__length - 1)]
        secondNumber = randint(0, self.__length - 1)
        while secondNumber == self.__representation[0]:
            secondNumber = randint(0, self.__length - 1)
        thirdNumber = randint(0, self.__length - 1)
        while thirdNumber == self.__representation[0] or thirdNumber == secondNumber:
            thirdNumber = randint(0, self.__length - 1)

        self.__representation.append(secondNumber)
        self.__representation.append(thirdNumber)

    @property
    def fitness(self):
        return self.__fitness

    @fitness.setter
    def fitness(self, newFitness):
        self.__fitness = newFitness

    @property
    def representation(self):
        return self.__representation

    @representation.setter
    def representation(self, newRepresentation):
        self.__representation = newRepresentation

    @property
    def length(self):
        return self.__length

    @length.setter
    def length(self, newLength):
        self.__length = newLength

    def crossover(self, chromosome):
        position1 = randint(0, 2)

        offSpring = ShortestPath(self.__graph)
        added = set()

        for index in range(0, position1):
            offSpring.representation[index] = self.__representation[index]
            added.add(self.__representation[index])

        toFill = position1
        for number in chromosome.representation:
            if number in added:
                continue
            if toFill >= 3:
                break
            offSpring.representation[toFill] = number
            toFill += 1

        return offSpring

    def mutation(self):
        probability = randint(0, 100)

        if probability <= 75:
            return

        low = randint(0, self.__length - 2)
        high = randint(low + 1, self.__length - 1)

        self.__representation[low:high] = self.__representation[low:high][::-1]

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
