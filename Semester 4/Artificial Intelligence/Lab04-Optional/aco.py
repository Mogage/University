import math

from ant import Ant


class ACO:
    def __init__(self, acoInfo):
        self.__acoInfo = acoInfo
        self.__ants = []
        self.__bestDistance = math.inf
        self.__bestTour = []
        self.__initialise()

    def __initialise(self):
        for _ in range(self.__acoInfo['colonySize']):
            self.__ants.append(Ant(self.__acoInfo['graph'], self.__acoInfo['alpha'], self.__acoInfo['beta']))

    def __addPheromone(self, tour, distance):
        toAdd = 1 / distance
        for count in range(self.__acoInfo['graph'].number_of_nodes() - 1):
            self.__acoInfo['graph'][tour[count]][tour[count + 1]]['pheromone'] *= (1.0 - self.__acoInfo['rho'])
            self.__acoInfo['graph'][tour[count]][tour[count + 1]]['pheromone'] += toAdd

    def __oneStep(self):
        for ant in self.__ants:
            self.__addPheromone(ant.findTour(), ant.getDistance())
            if ant.distance < self.__bestDistance:
                self.__bestTour = ant.tour
                self.__bestDistance = ant.distance
            #self.__addPheromone(self.__bestTour, self.__bestDistance)

    def run(self):
        for count in range(self.__acoInfo['numberOfSteps']):
            #self.__acoInfo['graph'].update_time(1)
            #print(f"step {count}")
            self.__oneStep()
        return {"Best Distance": self.__bestDistance, "Best Tour": self.__bestTour}
