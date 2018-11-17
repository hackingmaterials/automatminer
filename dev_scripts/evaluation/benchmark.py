import os
import time
import hashlib
import pprint
import copy

import git
from pymongo import MongoClient
from fireworks import FireTaskBase, Firework, Workflow, FWAction
from matminer.datasets.dataset_retrieval import load_dataset
from sklearn.metrics import f1_score, r2_score

import mslearn
from mslearn.featurization import AutoFeaturizer
from mslearn.preprocessing import DataCleaner, FeatureReducer
from mslearn.automl.adaptors import TPOTAdaptor
from mslearn.pipeline import MatPipe

# DB =  MongoClient("mongodb://%s:%s@ds111244.mlab.com:11244/automatminer" % (DB_USER, DB_PASSWORD)).automatminer
DB = MongoClient('localhost', 27017).automatminer
DB_USER = "miner"
DB_PASSWORD = "Materials2019"

DATASET_SET = ["elastic_tensor_2015"]
TARGETS = {"elastic_tensor_2015": "K_VRH"}
SCORING = {"elastic_tensor_2015": "r2"}
REWRITE_COLS = {"elastic_tensor_2015": {"formula": "composition"}}
RELEVANT_COLS = {"elastic_tensor_2015": ["K_VRH", "structure", "composition"]}


# todo: eventually this should use a test_idx so ensure that for every dataset for every repetition the same test set is used!

class RunPipe(FireTaskBase):
    _fw_name = 'RunPipe'

    def run_task(self, fw_spec):
        if fw_spec["learner_name"] == "TPOTAdaptor":
            learner = TPOTAdaptor
        else:
            raise ValueError("{} is an unknown learner name!"
                             "".format(self["learner_name"]))

        # Set up the pipeline and data
        pipe_config_dict = fw_spec["pipe_config"]
        pipe_config = {"learner": learner(**pipe_config_dict["learner_kwargs"]),
                       "reducer": FeatureReducer(
                           **pipe_config_dict["reducer_kwargs"]),
                       "cleaner": DataCleaner(
                           **pipe_config_dict["cleaner_kwargs"]),
                       "autofeaturizer_kwargs":
                           AutoFeaturizer(
                               **pipe_config_dict["autofeaturizer_kwargs"])}
        pipe = MatPipe(**pipe_config)
        dataset = fw_spec["dataset"]
        df = load_dataset(dataset)
        df = df.rename(columns=REWRITE_COLS[dataset])[RELEVANT_COLS[dataset]]
        target = TARGETS[dataset]

        # Run the benchmark
        t1 = time.time()
        predicted_test_df = pipe.benchmark(df, target, test_spec=0.2)
        elapsed_time = time.time() - t1

        # Save everything
        savedir = fw_spec["save_dir"]
        pipe.save(os.path.join(savedir, "pipe.p"))
        pipe.digest(os.path.join(savedir, "digest.txt"))
        predicted_test_df.to_csv(os.path.join(savedir, "test_df.csv"))
        pipe.post_fit_df.to_csv(os.path.join(savedir, "fitted_df.csv"))

        # Evaluate model
        true = predicted_test_df[target]
        test = predicted_test_df[target + " predicted"]
        if SCORING[dataset] == "r2":
            scorer = r2_score
        elif SCORING[dataset] == "f1":
            scorer = f1_score
        else:
            raise KeyError("Scoring {} not among valid options: [r2, f1].")
        score = scorer(true, test)

        # Extract important details for storage
        best_model = pipe.learner.best_models[0]
        features = pipe.learner.features
        n_features = len(features)
        n_test_samples_original = len(df[fw_spec["test_idx"]])

        pass_to_storage = {"score": score, "target": target,
                           "best_model": best_model,
                           "elapsed_time": elapsed_time,
                           "features": features,
                           "n_features": n_features,
                           "n_test_samples_original": n_test_samples_original,
                           "n_train_samples_original": len(
                               df) - n_test_samples_original,
                           "n_train_samples": len(pipe.post_fit_df),
                           "n_test_samples": len(test)
                           }
        return FWAction(update_spec=pass_to_storage)


class StorePipeResults(FireTaskBase):
    _fw_name = "StorePipeResults"

    def run_task(self, fw_spec):
        DB.pipes.insert_one(fw_spec)


class ConsolidateRuns(FireTaskBase):
    _fw_name = "ConsolidateRuns"


    def run_task(self, fw_spec):
        builds = DB.builds
        pipes = DB.pipes
        build_hash = fw_spec["build_hash"]

        tags_all = []
        performance_dict = {k: [] for k in DATASET_SET}
        features_dict = {k : [] for k in DATASET_SET}
        time_dict = {k: [] for k in DATASET_SET}

        for doc in pipes.find({'build_hash': build_hash}):
            tags_all.extend(doc["tags"])
            performance_dict[doc["dataset"]].append(doc["score"])
        pass
    #todo: finish this



def submit_build(launchpad, name, pipe_config, tags=None):
    top_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../..")
    repo = git.Repo(top_dir)
    last_commit = str(repo.head.commit)
    # Build hash is the combination of pipe configuration and current commit
    build_config_for_hash = copy.deepcopy(pipe_config)
    build_config_for_hash["last_commit"] = last_commit
    build_config_for_hash = str(build_config_for_hash).encode("UTF-8")
    build_hash = hashlib.sha1(build_config_for_hash).hexdigest()[:10]

    fws = []
    for dataset in DATASET_SET:
        spec = {"dataset": dataset,
                "pipe_config": pipe_config,
                "commit": last_commit,
                "name": name,
                "build_hash": build_hash,
                "tags": tags if tags else []}
        for trial in range(5):
            spec["trial"] = trial
            fws.append(Firework([RunPipe(), StorePipeResults()],
                                spec=spec,
                                name="{}: {} - trial {}".format(name, dataset,
                                                                trial)))

    # todo: link fws together and test
    wf = Workflow(fws, name="{}: build {}".format(name, build_hash))


if __name__ == "__main__":
    # mc = MongoClient("mongodb://%s:%s@ds111244.mlab.com:11244/automatminer" % (DB_USER, DB_PASSWORD))
    # print(mc.automatminer)

    submit_build("Test", {})

    # print(subprocess.check_output(["git", "describe"]).strip())
    # build = {"learner_name": "TPOTAdaptor",
    #          "learner_kwargs": {"max_time_mins": 2, "population_size": 10},
    #          "reducer_kwargs": None,
    #          "autofeaturizer_kwargs": None,
    #          "cleaner_kwargs": None,
    #          "build": None}
