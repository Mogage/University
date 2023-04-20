from random import randint
from math import sqrt
from math import inf

class Ant:
    def __init__(self, graph, alpha, beta):
        self.__alpha = alpha
        self.__beta = beta
        self.__graph = graph
        self.__tour = None
        self.__distance = 0

    @property
    def distance(self):
        return self.__distance

    @property
    def tour(self):
        return self.__tour

    def __getProbability(self, currentNode, nextNode, totalHeuristic, totalWeight):
        heuristicValue = pow(totalWeight / self.__graph[currentNode][nextNode]['weight'], self.__alpha)
        #heuristicValue = pow(totalWeight / self.__euclidianDistance(self.__graph[currentNode], self.__graph[nextNode]), self.__alpha)
        pheromoneTrace = pow(self.__graph[currentNode][nextNode]['pheromone'], self.__beta)
        return heuristicValue * pheromoneTrace / totalHeuristic

    def __getNextNode(self):
        currentNode = self.__tour[-1]
        unvisited = set(node for node in self.__graph.nodes() if node not in self.__tour)
        highestProbability = -1.0
        nextNode = None
        totalHeuristic = 0
        totalWeight = 0

        for node in unvisited:
            totalWeight += self.__graph[currentNode][node]['weight']
            #totalWeight += self.__euclidianDistance(self.__graph[currentNode], self.__graph[node])

        for node in unvisited:
            totalHeuristic += pow(self.__graph[currentNode][node]['pheromone'], self.__alpha) * \
                              pow(totalWeight / self.__graph[currentNode][node]['weight'], self.__beta)
                              #pow(totalWeight / self.__euclidianDistance(self.__graph[currentNode], self.__graph[node]), self.__beta)

        for node in unvisited:
            probability = self.__getProbability(currentNode, node, totalHeuristic, totalWeight)
            if probability >= highestProbability:
                highestProbability = probability
                nextNode = node

        return nextNode

    def findTour(self):
        # self.__tour = [randint(0, self.__graph.number_of_nodes() - 1)]
        self.__tour = [randint(1, self.__graph.number_of_nodes() - 1)]
        while len(self.__tour) < self.__graph.number_of_nodes():
            self.__tour.append(self.__getNextNode())
        return self.__tour


    ''' def __euclidianDistance(self, node1, node2):
        return sqrt((node1['x'] - node2['x']) ** 2 + (node1['y'] - node2['y']) ** 2)

    def getDistance(self):
        self.__distance = 0
        for count in range(self.__graph.number_of_nodes() - 1):
            self.__distance += self.__euclidianDistance(self.__graph[self.__tour[count]], self.__graph[self.__tour[count + 1]])

        self.__distance += self.__euclidianDistance(self.__graph[self.__tour[0]], self.__graph[self.__tour[-1]])
        return self.__distance'''


    def getDistance(self):
        self.__distance = 0
        for count in range(self.__graph.number_of_nodes() - 1):
            self.__distance += self.__graph[self.__tour[count]][self.__tour[count + 1]]['weight']

        self.__distance += self.__graph[self.__tour[0]][self.__tour[-1]]['weight']
        return self.__distance
