class Normalization:
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
