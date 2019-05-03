import math
import heapq
import random
from hashlib import sha1
from dataclasses import dataclass
from collections import OrderedDict, ChainMap

import numpy as np
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, f1_score, accuracy_score, roc_auc_score

from automatminer.automl.nn.wrapper import NNWrapper
from automatminer.base import LoggableMixin, DFMLAdaptor
from automatminer.utils.log import log_progress, AMM_LOG_FIT_STR, AMM_LOG_PREDICT_STR
from automatminer.utils.pkg import set_fitted, check_fitted
from automatminer.utils.ml import AMM_REG_NAME, AMM_CLF_NAME, regression_or_classification

__authors__ = ['Samy Cherfaoui <scherfaoui@lbl.gov>',
               'Alex Dunn <ardunn@lbl.gov']

param_grid = {
    "activation": ["sigmoid", "tanh", "relu", "elu"],
    "optimizers": ["sgd", "rmsprop", "adagrad", "adadelta", "nadam", "adamax", "adam"],
    "units": range(1, 1000),
    "hidden_layer_sizes": range(1, 5)
}


@dataclass(repr=True)
class NNModelInfo:
    params: dict
    gen: int
    score: (float, None) = None
    parents: (tuple, None) = None
    children: (list) = []

    def __lt__(self, other):
        return self.score < other.score

    @property
    def ordered_params(self) -> OrderedDict:
        return OrderedDict(self.params)

    @property
    def ref(self) -> str:
        model_hash = sha1(str(self.ordered_params).encode("UTF-8")).hexdigest()
        return "model_{}".format(model_hash)


def neg_mean_absolute_error(*args, **kwargs):
    return -1.0 * mean_absolute_error(*args, **kwargs)


def neg_mean_squared_error(*args, **kwargs):
    return -1.0 * mean_squared_error(*args, **kwargs)


class NNGA(DFMLAdaptor, LoggableMixin):
    def __init__(self, param_grid=param_grid, selection_rate=0.75, random_rate=0.05,
                 tournament_rate = 0.1, mutation_rate=0.05, elitism_rate=0.05, pop_size=15, reg_metric='mae',
                 clf_metric='f1', pbar=True):
        self.param_grid = param_grid
        self.selection_rate = selection_rate
        self.tournament_rate = tournament_rate
        self.random_rate = random_rate
        self.elitism_rate = elitism_rate
        self.mutation_rate = mutation_rate
        self.pop_size = 15
        self.pbar = pbar
        self.mode = None

        reg_metrics = {
            'neg_mae': neg_mean_absolute_error,
            'neg_mse': neg_mean_squared_error
        }

        clf_metrics = {
            "f1": f1_score,
            "roc_auc": roc_auc_score,
            "accuracy": accuracy_score
        }

        self.reg_scorer = reg_metrics[reg_metric]
        self.clf_scorer = clf_metrics[clf_metric]

        param_pop = [None] * pop_size
        for i in range(pop_size):
            params = {p: random.choice(g) for p, g in param_grid.items()}
            if any([p == params for p in param_pop]):
                continue
            else:
                param_pop[i] = params

        self.pop = [NNModelInfo(p, 0) for p in param_pop]
        self.model_class = NNWrapper

    @staticmethod
    def probability_to_number(n_individuals, probability):
        return math.ceil(n_individuals * probability)


    def tournament_select(self, individuals, n_winners, n_contestants):
        # choose k (the tournament size) individuals from the pop at random
        # choose the best individual from the tournament with probability p
        # choose the second best individual with probability p*(1-p)
        # choose the third best individual with probability p*((1-p)^2)
        # and so on

        p = self.selection_rate
        winners = [None] * n_winners
        for i in range(n_winners):
            t_pop = random.sample(individuals, n_contestants)
            t_pop_ranked = sorted(t_pop)
            t_pop_p = [None] * len(t_pop)
            for j, individual in enumerate(t_pop_ranked):
                individual.p_tournament = p * (1 - p) ** (j)
            winners[i] = random.choices(t_pop, weights=t_pop_p, k=1)
        return winners

    def evolve(self, gen, X_train, X_val, y_train, y_val):

        # evaluate all of this generations' population
        gen_pop = [i for i in self.pop if i.gen == gen]
        for individual in gen_pop:
            model = self.model_class(**individual.params)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            if self.mode == AMM_REG_NAME:
                scorer = self.reg_scorer
            elif self.mode == AMM_CLF_NAME:
                scorer = self.clf_scorer
            else:
                raise ValueError(
                    "'mode' attribute value {} is invalid! Must be either {} "
                    "(regression) or {} (classification)"
                    "".format(self.mode, AMM_REG_NAME, AMM_CLF_NAME)
                )
            score = scorer(y_val, y_pred)
            individual.score = score

        n_ran = self.probability_to_number(self.pop_size, self.random_rate)
        random_pop = random.sample(gen_pop, n_ran)
        n_elite = self.probability_to_number(self.pop_size, self.elitism_rate)
        elite_pop = heapq.nsmallest(n_elite, gen_pop)

        n_contestants = self.probability_to_number(self.pop_size, self.tournament_rate)
        n_winners = self.pop_size - n_ran - elite_pop
        tournament_pop = self.tournament_select(gen_pop, n_winners, n_contestants)

        pool = random_pop + elite_pop + tournament_pop

        # breed
        new_gen = [None] * self.pop_size
        for i in range(self.pop_size):
            # don't allow self-breeding
            parents = tuple(random.sample(pool, 2))
            m, f = parents

            child_params = {}
            for param in self.param_grid:
                if random.random < self.mutation_rate:
                    # mutate
                    child_params[param] = random.choice(self.param_grid[param])
                else:
                    # uniform crossover
                    child_params[param] = random.choice([m[param], f[param]])

            child = NNModelInfo(params=child_params, gen=gen + 1, parents=parents)
            f.children.append(child.ref)
            m.children.append(child.ref)
            new_gen[i] = child
        return new_gen


    @log_progress(AMM_LOG_FIT_STR)
    @set_fitted
    def fit(self, df, target, **kwargs):
        y = df[target].values
        X = df.drop(columns=target).values

        self.mode = regression_or_classification(df[target])

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.8)

        if self.mode == AMM_REG_NAME:
            pass

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
