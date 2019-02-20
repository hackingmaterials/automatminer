import random

class NeuralNetOptimizer():
    def __init__(self, X, y, model, retain = 5, random_select=0.05, mutate_chance=0.05, count=15):
        self.random_select = random_select
        self.mutate_chance = mutate_chance
        self.retain = retain
        self.X = X
        self.y = y
        self.Model = model
        ACTIVATION = ["sigmoid", "tanh", "relu", "elu"]
        OPTIMIZERS = ["sgd", "rmsprop", "adagrad", "adadelta", "nadam", "adamax", "adam"]
        self.NN_PARAM_CHOICES = {"units": range(1, 1000), "hidden_layer_sizes": range(1, 3), "optimizer": OPTIMIZERS,
                            "activation": ACTIVATION}
        pop = self.create_population(count)
        for _ in range(10):
            pop = self.evolve(pop, X, y)

        self.model_win = self.best_model(pop)

    def create_population(self, count):
        pop = []
        for _ in range(count):
            params = {"units": random.choice(self.NN_PARAM_CHOICES["units"]),
                      "hidden_layer_sizes": random.choice(self.NN_PARAM_CHOICES["hidden_layer_sizes"]),
                      "optimizer": random.choice(self.NN_PARAM_CHOICES["optimizer"]),
                      "activation": random.choice(self.NN_PARAM_CHOICES["activation"])}
            pop.append(params)
        return pop

    def breed(self, mother, father):
        children = []
        for _ in range(2):
            child = {}
            for param in self.NN_PARAM_CHOICES:
                child[param] = random.choice(
                    [mother[param], father[param]]
                )
            children.append(child)
        return children

    def mutate(self, network):
        mutation = random.choice(list(self.NN_PARAM_CHOICES.keys()))
        choice = random.choice(self.NN_PARAM_CHOICES[mutation])
        network[mutation] = random.choice(self.NN_PARAM_CHOICES[mutation])

    def evolve(self, pop, X, y):
        from sklearn.model_selection import KFold, cross_val_score
        import numpy as np

        models = [(self.Model(units=model["units"], hidden_layer_sizes=model["hidden_layer_sizes"],
                             optimizer=model["optimizer"], activation=model["activation"]), model) for model in pop]
        kfold = KFold(n_splits=2, shuffle=True, random_state=np.random.seed(7))
        grades = []
        if models[0][0].mode == "regression":
            for model, dicti in models:
                try:
                    score = cross_val_score(model, X, y, cv=kfold)
                    grades.append((score, dicti))
                except:
                    continue

        else:
            for model, dicti in models:
                score = cross_val_score(model, X, y, cv=kfold)
                if all([not np.isnan(result) for result in score]):
                    grades.append((score, dicti))
        grades = [x for x in sorted(grades, key=lambda x: x[0].mean(), reverse=True)]
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
        print(y)
        return parents

    def best_model(self, parents):
        params = parents[0]
        print(params)
        return self.Model(units=params["units"], hidden_layer_sizes=params["hidden_layer_sizes"],
                         optimizer=params["optimizer"], activation=params["activation"])



