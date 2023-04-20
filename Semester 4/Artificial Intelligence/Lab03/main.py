import os
import networkx as nx
import numpy as np
import math
import matplotlib.pyplot as plt
from random import seed, randint
from population import Population
from utils import generateNewValue


def fcEval(graph, x):
    # sphere function 
    # val = sum(xi ** 2 for xi in x)

    # Griewank function
    # term1 = sum(xi ** 2 / 4000 for xi in x)
    # cosinus = np.cos([xi for xi in x])
    # cosinus = [cosinus[i] / math.sqrt(i + 1) for i in range(len(x))]
    # term2 = np.prod([c for c in cosinus], axis = 0)
    # val = term1 - term2 + 1

    # Rastrigin function 
    val = 20 + sum(xi ** 2 - 10 * np.cos(2 * np.pi * xi) for xi in x)

    return val


def fitnessFunction2(graph, chromosomes):
    communityNumber = 1;
    communities = [0 for _ in range(graph.number_of_nodes() + 1)]
    with open("data_sets/karate.dat") as file:
        for line in file:
            aux = line.split(" ")
            numbers = []
            for x in aux:
                if x != '\n':
                    numbers.append(int(x))
            for x in numbers:
                communities[x] = communityNumber
            communityNumber += 1

    fitness = 0.0
    for count in range(graph.number_of_nodes()):
        fitness += abs(chromosomes[count] - communities[count] * count)

    return fitness


def fitnessFunction3(graph, chromosomes):
    communityNumber = 1;
    communities = [0 for _ in range(graph.number_of_nodes() + 1)]
    with open("data_sets/karate.dat") as file:
        for line in file:
            aux = line.split(" ")
            numbers = []
            for x in aux:
                if x != '\n':
                    numbers.append(int(x))
            for x in numbers:
                communities[x] = communityNumber
            communityNumber += 1

    real = fitnessFunction(graph, communities)
    fals = fitnessFunction(graph, chromosomes)

    return  real - fals

def fitnessFunction(graph, chromosome):
    m = 2 * graph.number_of_edges()
    Q = 0.0

    for count1 in range(0, graph.number_of_nodes()):
        for count2 in range(0, graph.number_of_nodes()):
            # We only add to the modularity if the nodes are in different communities
            if chromosome[count1] != chromosome[count2]:
                continue

            # Degree of node i
            k_i = graph.degree(count1)
            # Degree of node j
            k_j = graph.degree(count2)

            Q += graph.has_edge(count1, count2) - k_i * k_j / m

    return Q / m


def plotNetwork(network, communities):
    np.random.seed(1000)
    pos = nx.spring_layout(network)
    plt.figure(figsize=(16, 12))
    #nx.draw(network, pos, cmap=plt.get_cmap('Set2'), node_color=communities, with_labels=True)
    nx.draw_networkx_nodes(network, pos, node_size=150, cmap=plt.cm.RdYlBu, node_color = communities)
    nx.draw_networkx_edges(network, pos, alpha=0.3)
    plt.show()


def count_occurrences(numbers_list):
    counts = {}
    for num in numbers_list:
        if num in counts:
            counts[num] += 1
        else:
            counts[num] = 1
    return counts


def run():
    graph = nx.read_gml(f'data_sets/com-dblp.gml', label='id')
    noOfGenerations = 5
    populationSize = 5
    chromosomeSize = graph.number_of_nodes()

    seed(1)

    print("start prog")

    population = Population(populationSize, chromosomeSize, graph, fitnessFunction)

    print("evaluate pop 0")
    population.evaluate()

    print("start gen")

    for g in range(noOfGenerations):
        print("entry")
        population.oneGeneration()
        print("exit")

        bestChromosome = population.bestChromosome
        print('Best solution in generation ' + str(g) + ' is: x = ' + str(bestChromosome.representation) + ' f(x) = ' + str(bestChromosome.fitness))

    communities = population.bestChromosome.representation

    with open("output.txt", "w") as file:
        file.write(str(len(count_occurrences(communities))))
        file.write(str(count_occurrences(communities)))
        file.write(str(communities))

    print(communities)
    print(count_occurrences(communities))
    print(len(count_occurrences(communities)))
    # plotNetwork(graph, communities)


class Node:
    def __init__(self, id):
        self.__id = id

    @property
    def id(self):
        return self.__id

    @id.setter
    def id(self, newId):
        self.__id = newId

    def __str__(self):
        return "node\n[\nid "+str(self.__id)+"\n]\n"

class Edge:
    def __init__(self, source, target):
        self.__source = source
        self.__target = target

    @property
    def source(self):
        return self.__source

    @property
    def target(self):
        return self.__target

    @source.setter
    def source(self, newS):
        self.__source = newS

    @target.setter
    def target(self, newT):
        self.__target = newT

    def __str__(self):
        return "edge\n[\nsource "+str(self.__source)+"\ntarget "+str(self.__target)+"\n]\n"


if __name__ == "__main__":
    run()
    '''dic = set()
    inL = []
    out = []
    with open("data_sets/com-dblp.ungraph.txt", 'r') as file:
        for line in file:
            numbers = line.split("\t")
            dic.add(int(numbers[0]))
            dic.add(int(numbers[1]))
            inL.append(int(numbers[0]))
            out.append(int(numbers[1].replace('\n', '')))

    nodes = []
    new_nodes = {}

    index = 0

    for count in dic:
        if count not in new_nodes.keys():
            new_nodes[count] = index
            index += 1

    node = Node(0)
    edge = Edge(0, 0)

    print(len(dic))

    with open("data_sets/com-dblp.gml", "w") as file:
        file.write("graph[\n")
        for count in dic:
            node.id = new_nodes[count]
            file.write(str(node))
        for count in range(len(inL)):
            edge.source = new_nodes[inL[count]]
            edge.target = new_nodes[out[count]]
            file.write(str(edge))
        file.write("]")

    print(dic)'''
