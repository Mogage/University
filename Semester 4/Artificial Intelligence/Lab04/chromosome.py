from random import randint
from numpy import random


class Chromosome:
    def __init__(self, graph):
        self.__fitness = 0
        self.__length = 0
        self.__representation = []
        self.__graph = graph
        self.__initRepresentation(graph)

    def __initRepresentation(self, graph):
        '''added = set()
        self.__representation.append(node)
        added.add(node)
        while len(added) != graph.number_of_nodes():
            while True:
                randomNeighbor = randint(0, len(list(graph.neighbors(node))) - 1)
                neighbor = list(graph.neighbors(node))[randomNeighbor]
                if neighbor not in added:
                    added.add(neighbor)
                    self.__representation.append(neighbor)
                    node = neighbor
                    break'''
        self.__length = graph.number_of_nodes()
        self.__representation = random.permutation(self.__length)

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
        low = randint(0, self.__length - 2)
        high = randint(low + 1, self.__length)
        added = set()

        offSpring = Chromosome(self.__graph)

        for index in range(low, high):
            offSpring.representation[index] = self.__representation[index]
            added.add(self.__representation[index])

        toFill = high
        for index in range(high, chromosome.length):
            if chromosome.representation[index] in added:
                continue
            if toFill >= chromosome.length:
                toFill = 0
                break
            offSpring.representation[toFill] = chromosome.representation[index]
            toFill += 1

        for index in range(0, high):
            if chromosome.representation[index] in added:
                continue
            if toFill >= chromosome.length:
                toFill = 0
            offSpring.representation[toFill] = chromosome.representation[index]
            toFill += 1

        return offSpring

        '''size = min(self.__length, chromosome.length)
        equals = []
        for count in range(size):
            if self.__representation[count] == chromosome.representation[count]:
                equals.append(count)
        toChangeFrom = randint(0, len(equals) - 1)

        offSpring1 = Chromosome(self.__graph, self.__node)
        offSpring2 = Chromosome(self.__graph, self.__node)
        offSpring1.representation = self.__representation[:equals[toChangeFrom]] + chromosome.representation[equals[toChangeFrom]:]
        offSpring2.representation = chromosome.representation[:equals[toChangeFrom]] + self.__representation[equals[toChangeFrom]:]
        offSpring1.length = len(offSpring1.representation)
        offSpring2.length = len(offSpring2.representation)

        return offSpring1, offSpring2'''

    def mutation(self):
        low = randint(0, self.__length - 2)
        high = randint(low + 1, self.__length - 1)

        self.__representation[low:high] = self.__representation[low:high][::-1]

        '''startPoint = randint(0, self.__length - 2)
        added = set()
        newRepresentation = []
        for count in range(startPoint + 1):
            if self.__representation[count] in added:
                startPoint = count
                break
            newRepresentation.append(self.__representation[count])
            added.add(self.__representation[count])
        node = self.__representation[startPoint]

        while len(added) != self.__length:

            bestNeighbor = 0
            bestNeighborWeight = 100000000
            for neighbor in self.__graph.neighbors(node):
                if self.__graph[node][neighbor]['weight'] < bestNeighborWeight and neighbor not in added:
                    bestNeighborWeight = self.__graph[node][neighbor]['weight']
                    bestNeighbor = neighbor
            added.add(bestNeighbor)
            newRepresentation.append(bestNeighbor)

            while True:
                randomNeighbor = randint(0, len(list(self.__graph.neighbors(node))) - 1)
                neighbor = list(self.__graph.neighbors(node))[randomNeighbor]
                if neighbor not in added:
                    added.add(neighbor)
                    newRepresentation.append(neighbor)
                    node = neighbor
                    break

        self.__representation = newRepresentation'''

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
