import math
import heapq
import random
from hashlib import sha1
from typing import List
from dataclasses import dataclass, field
from collections import OrderedDict, ChainMap

import numpy as np
from tqdm import tqdm
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, f1_score, accuracy_score, roc_auc_score

from automatminer.automl.nn.wrapper import NNWrapper
from automatminer.base import LoggableMixin, DFMLAdaptor
from automatminer.utils.log import log_progress, AMM_LOG_FIT_STR, AMM_LOG_PREDICT_STR
from automatminer.utils.pkg import set_fitted, check_fitted
from automatminer.utils.ml import AMM_REG_NAME, AMM_CLF_NAME, regression_or_classification

__authors__ = ['Alex Dunn <ardunn@lbl.gov',
               'Samy Cherfaoui <scherfaoui@lbl.gov>']

param_grid = {
    "activation": ["sigmoid", "tanh", "relu", "elu"],
    "optimizer": ["sgd", "rmsprop", "adagrad", "adadelta", "nadam", "adamax", "adam"],
    "units": range(1, 5),
    "hidden_layer_sizes": range(1, 3)
}


@dataclass(repr=True)
class NNModelInfo:
    params: dict
    gen: int
    score: (float, None) = None
    parents: (tuple, None) = None
    children: List[int] = field(default_factory=list)

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
                 tournament_rate = 0.1, mutation_rate=0.05, elitism_rate=0.05, pop_size=15, reg_metric='neg_mae',
                 clf_metric='f1', generations=10, pbar=True, logger=True):
        self.param_grid = param_grid
        self.selection_rate = selection_rate
        self.tournament_rate = tournament_rate
        self.random_rate = random_rate
        self.elitism_rate = elitism_rate
        self.mutation_rate = mutation_rate
        self.pop_size = 15
        self.pbar = pbar
        self.mode = None
        self.generations = generations
        self.best_individual = None
        self.best_model = None
        self._logger = self.get_logger(logger)


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
            for j, _ in enumerate(t_pop_ranked):
                t_pop_p = p * (1 - p) ** (j)
            winners[i] = random.choices(t_pop, weights=t_pop_p, k=1)
        return winners

    def evolve(self, gen, X_train, X_val, y_train, y_val):
        # evaluate all of this generations' population

        print("in evolve: generation is", gen)

        gen_pop = [i for i in self.pop if i.gen == gen]

        print("gen pop is:", gen_pop)
        for individual in gen_pop:
            print("individual is", individual)
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
        ran_pop = random.sample(gen_pop, n_ran)
        n_elite = self.probability_to_number(self.pop_size, self.elitism_rate)
        elite_pop = heapq.nsmallest(n_elite, gen_pop)

        n_contestants = self.probability_to_number(self.pop_size, self.tournament_rate)
        n_winners = self.pop_size - n_ran - n_elite
        tournament_pop = self.tournament_select(gen_pop, n_winners, n_contestants)

        pool = ran_pop + elite_pop + tournament_pop

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

        for i in tqdm(range(self.generations), desc="NN Generation"):
            splits = train_test_split(X, y, test_size=0.8)
            new_gen = self.evolve(i, *splits)
            self.pop.extend(new_gen)

        self.best_individual = min(self.pop)
        self.logger(self._log_prefix + "Best model found: {}".format(self.best_individual))
        self.logger(self._log_prefix + "Best model training: {}".format(self.best_individual))
        self.best_model = self.model_class(**self.best_individual["params"])
        self.best_model.fit(X, y)


    @log_progress(AMM_LOG_PREDICT_STR)
    @check_fitted
    def predict(self, df, target):
        X = df[self._features].values
        y_pred = self.best_pipeline.predict(X)
        df[target + " predicted"] = y_pred
        return df

    @property
    @check_fitted
    def best_pipeline(self):
        return self.best_model

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
        return self.best_model


if __name__ == "__main__":
    from sklearn.datasets import load_boston
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_absolute_error
    import pandas as pd

    target = "PRICE"
    boston = load_boston()
    df = pd.DataFrame(boston.data)
    df.columns = boston.feature_names
    df[target] = boston.target

    df_train, df_test = train_test_split(df)

    nnga = NNGA()
    nnga.fit(df, "PRICE")
    df_pred = nnga.predict(df_train, target)

    print(df)

    predictions = df_pred[target + " predicted"]
    true = df_test[target]

    print(mean_absolute_error(y_true=true, y_pred=predictions))

    print(nnga.best_model.model_.summary())
