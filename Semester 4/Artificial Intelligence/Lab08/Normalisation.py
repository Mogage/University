from numpy import std


class CustomNormalisation:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, data):
        length = 0
        numbersSum = 0
        stdSum = 0
        for sample in data:
            for number in sample:
                numbersSum += number
                length += 1
        self.mean = numbersSum / length
        for sample in data:
            for number in sample:
                stdSum += (number - self.mean) ** 2
        self.std = (1 / length * stdSum) ** 0.5

    def transform(self, data):
        return [[(x - self.mean) / self.std for x in sample] for sample in data]


class Normalisation:
    @staticmethod
    def __scale(data):
        minFeature = min(data)
        maxFeature = max(data)
        return [(x - minFeature) / (maxFeature - minFeature) for x in data]

    @staticmethod
    def __zeroCentralisation(data):
        mean = sum(data) / len(data)
        return [x - mean for x in data]

    @staticmethod
    def __statisticalNormalization(data):
        size = len(data)
        mean = sum(data) / size
        # stdCustom = (1 / size * sum([(x - mean) ** 2 for x in data])) ** 0.5
        stdCustom = 1 if std(data) == 0 else std(data)
        return [(x - mean) / stdCustom for x in data]

    def __normalizeList(self, data, ntype):
        if ntype == 'statistical':
            return [self.__statisticalNormalization(x) for x in data]
        elif ntype == 'scale':
            return [self.__scale(x) for x in data]
        elif ntype == 'zero':
            return [self.__zeroCentralisation(x) for x in data]
        else:
            raise Exception('Invalid self type')

    def __normalizeSimple(self, data, ntype):
        if ntype == 'statistical':
            return self.__statisticalNormalization(data)
        elif ntype == 'scale':
            return self.__scale(data)
        elif ntype == 'zero':
            return self.__zeroCentralisation(data)
        else:
            raise Exception('Invalid self type')

    def normalise(self, data, ntype='statistical'):
        if isinstance(data, dict) and len(data.keys()) > 1:
            return self.__normalizeList(data.values(), ntype)
        if isinstance(data[0], list):
            return self.__normalizeList(data, ntype)
        return self.__normalizeSimple(data, ntype)
