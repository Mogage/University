from random import randint
from chromosome import Chromosone


class Population():
    def __init__(self, populationSize, noOfNodes, graph, evaluateFunction) -> None:
        self.__populationSize = populationSize
        self.__population = []
        self.__graph = graph
        self.__evaluateFunction = evaluateFunction
        for _ in range(populationSize):
            self.__population.append(Chromosone(noOfNodes, 2))

    @property
    def population(self):
        return self.__population

    @property
    def bestChromosome(self):
        best = self.__population[0]
        for chromosome in self.__population:
            if chromosome.fitness < best.fitness:
                best = chromosome
        return best
    
    def __getWorstChromosomeIndex(self):
        best = 0
        for count in range(self.__populationSize):
            if self.__population[count].fitness > self.__population[best].fitness:
                best = count
        return best

    @property
    def worstChromosome(self):
        return self.__population[self.__getWorstChromosomeIndex()]
    
    @worstChromosome.setter
    def worstChromosome(self, newChromosome):
        self.__population[self.__getWorstChromosomeIndex()] = newChromosome

    def selection(self):
        pos1 = randint(0, self.__populationSize - 1)
        pos2 = randint(0, self.__populationSize - 1)
        if self.__population[pos1].fitness < self.__population[pos2].fitness:
            return self.__population[pos1]
        else:
            return self.__population[pos2]

    def evaluate(self):
        for population in range(self.__populationSize):
            self.__population[population].fitness = self.__evaluateFunction(self.__graph, self.__population[population].representation)

    def oneGeneration(self):
        newPop = []
        print("new Gen")
        for _ in range(self.__populationSize):
            p1 = self.selection()
            p2 = self.selection()
            off = p1.crossover(p2)
            off.mutation()
            newPop.append(off)
        self.__population = newPop
        print("end new Gen")
        self.evaluate()
        print("Evaluated new gen")

    '''def oneGeneration(self):
        for _ in range(self.__populationSize):
            parent1 = self.selection()
            parent2 = self.selection()

            off = parent1.crossover(parent2)
            off.mutation()
            off.fitness = self.__evaluateFunction(self.__graph, off.representation)
            if off > self.worstChromosome:
                self.worstChromosome = off
    '''