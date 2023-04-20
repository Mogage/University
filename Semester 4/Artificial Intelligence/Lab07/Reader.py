import csv


class Reader:
    def __init__(self, filePath):
        self.__filePath = filePath
        self.__dataNames = []
        self.__data = []
        self.__loadData()

    @property
    def dataNames(self):
        return self.__dataNames

    @property
    def data(self):
        return self.__data

    def __loadData(self):
        with open(self.__filePath) as csv_file:
            csvReader = csv.reader(csv_file, delimiter=',')
            self.__dataNames = next(csvReader)
            for row in csvReader:
                self.__data.append(row)

    def extractFeatures(self, features, output=False):
        results = []
        indexes = []
        for feature in features:
            indexes.append(self.__dataNames.index(feature))

        if output:
            return [float(row[indexes[0]]) for row in self.__data]

        for row in self.__data:
            listToAppend = []
            for index in indexes:
                listToAppend.append(float(row[index]))
            results.append(listToAppend)

        return results
