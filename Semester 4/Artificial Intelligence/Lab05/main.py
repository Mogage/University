from math import sqrt, log2, log


def calculateError(dic, noOfValues):
    MAE = []
    MSE = []
    RMSE = []
    totalMAE = 0
    totalMSE = 0
    totalRMSE = 0
    for index in range(noOfValues):
        MAE.append(
            sum(abs(real - computed) for real, computed in zip(dic[index], dic[index + noOfValues])) / len(dic[index]))
        MSE.append(sum((real - computed) ** 2 for real, computed in zip(dic[index], dic[index + noOfValues])) / len(
            dic[index]))
        RMSE.append(
            sqrt(sum((real - computed) ** 2 for real, computed in zip(dic[index], dic[index + noOfValues]))) / len(
                dic[index]))
        totalMAE += sum(abs(real - computed) for real, computed in zip(dic[index], dic[index + noOfValues])) / len(
            dic[index])
        totalMSE += sum((real - computed) ** 2 for real, computed in zip(dic[index], dic[index + noOfValues])) / len(
            dic[index])
        totalRMSE += sqrt(
            sum((real - computed) ** 2 for real, computed in zip(dic[index], dic[index + noOfValues]))) / len(
            dic[index])

    partDic = {'MAE': MAE, 'MSE': MSE, 'RMSE': RMSE}
    totalDic = {'total MAE': totalMAE, 'total MSE': totalMSE, 'total RMSE': totalRMSE}

    return partDic, totalDic


def getComputedLabels(labelNames, computedOutputs):
    computedLabels = []
    for p in computedOutputs:
        probMaxPos = p.index(max(p))
        label = labelNames[probMaxPos]
        computedLabels.append(label)
    return computedLabels


def evalClassification(realLabels, computedLabels, labelNames, probabilities=False):
    if probabilities:
        computedLabels = getComputedLabels(labelNames, computedLabels)

    accuracy = sum(1 if real == computed else 0 for real, computed in zip(realLabels, computedLabels)) / len(realLabels)
    precision = {}
    recall = {}

    for label in labelNames:
        precision[label] = 0
        recall[label] = 0
        TP = 0
        FP = 0
        FN = 0
        for real, computed in zip(realLabels, computedLabels):
            if real == label:
                if computed == label:
                    TP += 1
                else:
                    FN += 1
            else:
                if computed == label:
                    FP += 1

        precision[label] = TP / (TP + FP)
        recall[label] = TP / (TP + FN)

    return accuracy, precision, recall


def readCSVFile(fileName, intValues=False):
    with open(fileName, 'r') as f:
        words = f.readline().strip().split(',')
        dic = {}
        for index in range(len(words)):
            dic[index] = []
        for line in f:
            values = line.strip().split(',')
            for index in range(len(values)):
                if intValues:
                    dic[index].append(int(values[index]))
                else:
                    dic[index].append(values[index])

    return dic


def crossEntropy(trueFileName, probabilitiesFileName, type='binary'):
    crossEntropyValue = 0
    with open(trueFileName, 'r') as trueFile:
        with open(probabilitiesFileName, 'r') as probabilitiesFileNam:
            trueElements = trueFile.readline().strip().split(' ')
            probabilitiesElements = probabilitiesFileNam.readline().strip().split(' ')
            size = len(trueElements)
            for index in range(size):
                if type == 'binary':
                    crossEntropyValue += float(trueElements[index]) * log2(float(probabilitiesElements[index]))
                elif type == 'multi-class':
                    crossEntropyValue += float(trueElements[index]) * log(float(probabilitiesElements[index]))
                elif type == 'multi-target':
                    crossEntropyValue += (float(trueElements[index]) * log(float(probabilitiesElements[index])) + (
                                1 - float(trueElements[index])) * log(1 - float(probabilitiesElements[index])))
    return -crossEntropyValue


def run():
    dic = readCSVFile('data/sport.csv', True)
    print(calculateError(dic, 3))
    dic = readCSVFile('data/flowers.csv', False)
    print(evalClassification(dic[0], dic[1], ['Daisy', 'Rose', 'Tulip']))
    print(crossEntropy('data/true-binary.txt', 'data/probabilities-binary.txt', 'binary'))
    print(crossEntropy('data/true-multi-class.txt', 'data/probabilities-multi-class.txt', 'multi-class'))
    print(crossEntropy('data/true-multi-target.txt', 'data/probabilities-multi-target.txt', 'multi-target'))


if __name__ == '__main__':
    run()
