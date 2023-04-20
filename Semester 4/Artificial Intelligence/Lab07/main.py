from Reader import Reader
from Solver import Solver


if __name__ == "__main__":
    filePath = 'data/world-happiness-report-2017.csv'
    reader = Reader(filePath)
    dataNames = reader.dataNames
    data = reader.data
    inputs = reader.extractFeatures(['Economy..GDP.per.Capita.', 'Freedom'])
    output = reader.extractFeatures(['Happiness.Score'], True)

    solver = Solver(inputs, output, True)
    solver.splitTrainTest()
    solver.train()
    solver.test()
