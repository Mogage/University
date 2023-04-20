from solver import Solver
import csv

def load_data(file_path, columns):
    results = {}
    indexes = {}

    with open(file_path) as csvFile:
        headers = csvFile.readline().strip().split(',')
        for header in headers:
            if header in columns:
                results[header] = []
                indexes[header] = headers.index(header)
        for line in csvFile:
            line = line.strip().split(',')
            for column in columns:
                results[column].append(float(line[indexes[column]]))

    return results


if __name__ == '__main__':
    filePath = 'data/v3_world-happiness-report-2017.csv'

    gdp = 'Economy..GDP.per.Capita.'
    freedom = 'Freedom'
    happiness = 'Happiness.Score'
    columns = [happiness, gdp, freedom]
    data = load_data(filePath, columns)
    solver = Solver(data, columns)
    solver.split_dataset_train_test()
    solver.train()
    solver.test()
