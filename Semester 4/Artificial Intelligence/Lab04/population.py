from chromosome import Chromosome
from shortestPath import ShortestPath


class Population:
    def __init__(self, populationParams):
        self.__fitnessFunction = populationParams["fitnessFunction"]
        self.__graph = populationParams["graph"]
        self.__populationSize = populationParams["populationSize"]
        self.__population = []

    @property
    def population(self):
        return self.__population

    @property
    def bestChromosomes(self):
        best = self.__population[0]
        bests = []
        for chromosome in self.__population:
            if chromosome.fitness < best.fitness:
                best = chromosome
                bests = [chromosome]
            elif chromosome.fitness == best.fitness:
                bests.append(chromosome)
        return bests

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

    def initialisation(self):
        for _ in range(self.__populationSize):
            self.__population.append(Chromosome(self.__graph))
            #self.__population.append(ShortestPath(self.__graph))

    def evaluation(self):
        for population in range(self.__populationSize):
            self.__population[population].fitness = \
                self.__fitnessFunction(self.__graph, self.__population[population].representation)

    def selectParents(self):
        parent1, parent2 = self.__population[0], self.__population[0]
        for individual in self.__population:
            if individual > parent1:
                parent2 = parent1
                parent1 = individual
            elif individual > parent2:
                parent2 = individual

        return parent1, parent2

    def oneGeneration(self):
        for _ in range(self.__populationSize):
            parent1, parent2 = self.selectParents()

            offspring = parent1.crossover(parent2)

            offspring.mutation()
            offspring.fitness = self.__fitnessFunction(self.__graph, offspring.representation)

            if offspring < self.worstChromosome:
                self.worstChromosome = offspring

            '''offspring1, offspring2 = parent1.crossover(parent2)

            offspring1.mutation()
            offspring2.mutation()

            offspring1.fitness = self.__fitnessFunction(self.__graph, offspring1.representation)
            offspring2.fitness = self.__fitnessFunction(self.__graph, offspring2.representation)

            if offspring1 < self.worstChromosome:
                self.worstChromosome = offspring1
            if offspring2 < self.worstChromosome:
                self.worstChromosome = offspring2'''
