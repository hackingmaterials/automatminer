import random

import numpy as np
from sklearn.model_selection import KFold, cross_val_score, train_test_split

from automatminer.automl.nn.wrapper import NNWrapper
from automatminer.base import LoggableMixin, DFMLAdaptor
from automatminer.utils.log import log_progress, AMM_LOG_FIT_STR, AMM_LOG_PREDICT_STR
from automatminer.utils.pkg import set_fitted, check_fitted
from automatminer.utils.ml import AMM_REG_NAME, AMM_CLF_NAME

__authors__ = ['Samy Cherfaoui <scherfaoui@lbl.gov>',
               'Alex Dunn <ardunn@lbl.gov']

param_grid = {
    "activation": ["sigmoid", "tanh", "relu", "elu"],
    "optimizers": ["sgd", "rmsprop", "adagrad", "adadelta", "nadam", "adamax", "adam"]
    "units": range(1, 1000),
    "hidden_layer_sizes": range(1, 5)
}


class NNGA(DFMLAdaptor, LoggableMixin):

    def __init__(self, param_grid=param_grid, retain=5, random_select=0.05,
                 mutation_rate=0.05, pop_size=15, pbar=True):
        self.param_grid = param_grid
        self.random_select = random_select
        self.mutation_rate = mutation_rate
        self.pop_size = 15
        self.pbar = pbar

        pop = [None] * pop_size
        for i in range(pop_size):
            params = {p: random.choice(g) for p, g in param_grid.items()}
            pop[i] = params
        self.pop = pop
        self.model_class = NNWrapper

    def mutate(self, pop):
        pass

    def breed(self, mom, dad):
        pass

    @log_progress(AMM_LOG_FIT_STR)
    @set_fitted
    def fit(self, df, target, **kwargs):
        y = df[target].values.tolist()
        X = df.drop(columns=target).values.tolist()
        models = [self.model_class(**p) for p in self.pop]
        train, validation = train_test_split(X, y)

        y = df[target].values.tolist()
        X = df.drop(columns=target).values.tolist()

        kfold = KFold(n_splits=2, shuffle=True, random_state=np.random.seed(7))
        grades = []
        if models[0][0].mode == AMM_REG_NAME:
            for model, dicti in models:
                score = cross_val_score(model, X, y, cv=kfold)
                grades.append((score, dicti))
        else:
            for model, dicti in models:
                score = cross_val_score(model, X, y, cv=kfold)
                if all([not np.isnan(result) for result in score]):
                    grades.append((score, dicti))
        grades = [x for x in
                  sorted(grades, key=lambda x: x[0].mean(), reverse=True)]
        retain_length = int(len(grades) * self.retain)
        parents = [dictionary for model, dictionary in grades[:retain_length]]
        for individual, dicti in grades[retain_length:]:
            if self.random_select > random.random():
                parents.append(dicti)
        for individual in parents:
            if self.mutate_chance > random.random():
                self.mutate(individual)
        parents_length = len(parents)
        desired_length = len(pop) - parents_length
        children = []
        while len(children) < desired_length:
            male = random.randint(0, parents_length - 1)
            female = random.randint(0, parents_length - 1)
            if male != female:
                male = parents[male]
                female = parents[female]
                babies = self.breed(male, female)
                for baby in babies:
                    if len(children) < desired_length:
                        children.append(baby)
        parents.extend(children)
        return parents

    @log_progress(AMM_LOG_PREDICT_STR)
    @check_fitted
    def predict(self, df, target):
        pass

    @property
    @check_fitted
    def best_pipeline(self):
        return self._backend.fitted_pipeline_

    @property
    @check_fitted
    def features(self):
        return self._features

    @property
    @check_fitted
    def ml_data(self):
        return self._ml_data

    @property
    @check_fitted
    def backend(self):
        return self._backend


# def __init__(self, X, y, model, retain=5, random_select=0.05,
#              mutate_chance=0.05, pop_size=15):
#     self.random_select = random_select
#     self.mutate_chance = mutate_chance
#     self.retain = retain
#     self.X = X
#     self.y = y
#     self.model = model
#     activation = ["sigmoid", "tanh", "relu", "elu"]
#     optimizers = ["sgd", "rmsprop", "adagrad", "adadelta", "nadam",
#                   "adamax", "adam"]
#     self.param_grid = {"units": range(1, 1000),
#                        "hidden_layer_sizes": range(1, 3),
#                        "optimizer": optimizers,
#                        "activation": activation}
#
#     population = [None] * pop_size
#     for i in range(pop_size):
#         params = {
#             "units": random.choice(self.param_grid["units"]),
#             "hidden_layer_sizes": random.choice(
#                 self.param_grid["hidden_layer_sizes"]),
#             "optimizer": random.choice(self.param_grid["optimizer"]),
#             "activation": random.choice(self.param_grid["activation"])}
#         population.append(params)
#
#     self.population_size = population
#     self.model = None
#
# def optimize(self):
#     for _ in range(10):
#         pop = self.evolve(pop, self.X, self.y)
#     self.model = self.best_model(self.population_size)
#
# def breed(self, mother, father):
#     children = []
#     for _ in range(2):
#         child = {}
#         for param in self.param_grid:
#             child[param] = random.choice([mother[param], father[param]])
#         children.append(child)
#     return children
#
# def mutate(self, network):
#     mutation = random.choice(list(self.param_grid.keys()))
#     choice = random.choice(self.param_grid[mutation])
#     network[mutation] = choice
#
# def evolve(self, pop, X, y):
#     models = [
#         (self.model(units=model_dict["units"],
#                           hidden_layer_sizes=model_dict["hidden_layer_sizes"],
#                           optimizer=model_dict["optimizer"],
#                           activation=model_dict["activation"]), model_dict) for model_dict
#               in pop]
#     kfold = KFold(n_splits=2, shuffle=True, random_state=np.random.seed(7))
#     grades = []
#     if models[0][0].mode == AMM_REG_NAME:
#         for model, dicti in models:
#             score = cross_val_score(model, X, y, cv=kfold)
#             grades.append((score, dicti))
#     else:
#         for model, dicti in models:
#             score = cross_val_score(model, X, y, cv=kfold)
#             if all([not np.isnan(result) for result in score]):
#                 grades.append((score, dicti))
#     grades = [x for x in sorted(grades, key=lambda x: x[0].mean(), reverse=True)]
#     retain_length = int(len(grades) * self.retain)
#     parents = [dictionary for model, dictionary in grades[:retain_length]]
#     for individual, dicti in grades[retain_length:]:
#         if self.random_select > random.random():
#             parents.append(dicti)
#     for individual in parents:
#         if self.mutate_chance > random.random():
#             self.mutate(individual)
#     parents_length = len(parents)
#     desired_length = len(pop) - parents_length
#     children = []
#     while len(children) < desired_length:
#         male = random.randint(0, parents_length - 1)
#         female = random.randint(0, parents_length - 1)
#         if male != female:
#             male = parents[male]
#             female = parents[female]
#             babies = self.breed(male, female)
#             for baby in babies:
#                 if len(children) < desired_length:
#                     children.append(baby)
#     parents.extend(children)
#     return parents
#
# def best_model(self, parents):
#     params = parents[0]
#     return self.model(units=params["units"],
#                       hidden_layer_sizes=params["hidden_layer_sizes"],
#                       optimizer=params["optimizer"],
#                       activation=params["activation"])


if __name__ == "__main__":
    nnp = NNGA()
