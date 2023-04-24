from Reader import Reader
from Solver import Solver
from sklearn.datasets import load_linnerud
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from numpy import std
from numpy import mean
from numpy import absolute
from NormalizationFactory import NormalizationFactory as Normalization


def runMultiTarget():
    X, y = load_linnerud(return_X_y=True)

    normalization = Normalization()

    X = normalization.normalize(X, 'statistical')
    y = normalization.normalize(y, 'statistical')

    # model = DecisionTreeRegressor()
    model = KNeighborsRegressor()
    wrapper = MultiOutputRegressor(model)

    cv = RepeatedKFold(n_splits=10, n_repeats=5, random_state=1)
    n_scores = cross_val_score(wrapper, X, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
    n_scores = absolute(n_scores)

    print('MAE: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))


def runSingleTarget():
    filePath = 'data/world-happiness-report-2017.csv'
    reader = Reader(filePath)
    inputs = reader.extractFeatures(['Economy..GDP.per.Capita.', 'Freedom'])
    output = reader.extractFeatures(['Happiness.Score'], True)

    solver = Solver(inputs, output, True)
    solver.splitTrainTest()
    solver.train()
    solver.test()


if __name__ == "__main__":
    # runSingleTarget()
    runMultiTarget()
