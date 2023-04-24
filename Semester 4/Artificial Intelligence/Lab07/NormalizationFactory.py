from numpy import std


class NormalizationFactory:
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
        stdCustom = std(data)
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

    def normalize(self, data, ntype='statistical'):
        if isinstance(data, dict) and len(data.keys()) > 1:
            return self.__normalizeList(data.values(), ntype)
        if isinstance(data[0], list):
            return self.__normalizeList(data, ntype)
        return self.__normalizeSimple(data, ntype)
