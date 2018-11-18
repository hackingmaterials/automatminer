import os
import time
import hashlib
import pprint
import copy
import datetime

import numpy as np
import git
from pymongo import MongoClient
from fireworks import FireTaskBase, Firework, Workflow, FWAction, LaunchPad, explicit_serialize
from fireworks.core.rocket_launcher import launch_rocket
from matminer.datasets.dataset_retrieval import load_dataset
from sklearn.metrics import f1_score, r2_score

from mslearn.featurization import AutoFeaturizer
from mslearn.preprocessing import DataCleaner, FeatureReducer
from mslearn.automl.adaptors import TPOTAdaptor
from mslearn.pipeline import MatPipe

# DB_USER =
# DB_PASSWORD =
# DB =  MongoClient("mongodb://%s:%s@ds111244.mlab.com:11244/automatminer" % (DB_USER, DB_PASSWORD)).automatminer
DB = MongoClient('localhost', 27017).automatminer

N_TRIALS = 5
DATASET_SET = ["elastic_tensor_2015"]
TARGETS = {"elastic_tensor_2015": "K_VRH"}
SCORING = {"elastic_tensor_2015": "r2"}
REWRITE_COLS = {"elastic_tensor_2015": {"formula": "composition"}}
RELEVANT_COLS = {"elastic_tensor_2015": ["K_VRH", "structure", "composition"]}

# todo: eventually this should use a test_idx so ensure that for every dataset for every repetition the same test set is used!


@explicit_serialize
class RunPipe(FireTaskBase):
    _fw_name = 'RunPipe'
    def run_task(self, fw_spec):
        pipe_config_dict = fw_spec["pipe_config"]

        if pipe_config_dict["learner_name"] == "TPOTAdaptor":
            learner = TPOTAdaptor
        else:
            raise ValueError("{} is an unknown learner name!"
                             "".format(self["learner_name"]))

        # Set up the pipeline and data
        pipe_config = {"learner": learner(**pipe_config_dict["learner_kwargs"]),
                       "reducer": FeatureReducer(
                           **pipe_config_dict["reducer_kwargs"]),
                       "cleaner": DataCleaner(
                           **pipe_config_dict["cleaner_kwargs"]),
                       "autofeaturizer":
                           AutoFeaturizer(
                               **pipe_config_dict["autofeaturizer_kwargs"])}
        pipe = MatPipe(**pipe_config)
        dataset = fw_spec["dataset"]
        df = load_dataset(dataset).iloc[:100]
        df = df.rename(columns=REWRITE_COLS[dataset])[RELEVANT_COLS[dataset]]
        target = TARGETS[dataset]

        # Run the benchmark
        t1 = time.time()
        predicted_test_df = pipe.benchmark(df, target, test_spec=0.2)
        elapsed_time = time.time() - t1

        # Save everything
        savedir = fw_spec["save_dir"]
        if not os.path.exists(savedir):
            os.makedirs(savedir)
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
        if pipe_config_dict["learner_name"] == "TPOTAdaptor":
            best_pipeline = [str(step) for step in pipe.learner.backend.fitted_pipeline_.steps]
        else:
            raise ValueError("Other backends are not supported yet!!")
        features = pipe.learner.features
        n_features = len(features)
        n_test_samples_original = len(df) * 0.2

        pass_to_storage = {"score": score, "target": target,
                           "best_pipeline": best_pipeline,
                           "elapsed_time": elapsed_time,
                           "features": features,
                           "n_features": n_features,
                           "n_test_samples_original": n_test_samples_original,
                           "n_train_samples_original": len(
                               df) - n_test_samples_original,
                           "n_train_samples": len(pipe.post_fit_df),
                           "n_test_samples": len(test),
                           "test_sample_frac_retained": len(
                               test) / n_test_samples_original,
                           "completion_time": datetime.datetime.utcnow()
                           }
        return FWAction(update_spec=pass_to_storage)

@explicit_serialize
class StorePipeResults(FireTaskBase):
    _fw_name = "StorePipeResults"

    def run_task(self, fw_spec):
        DB.pipes.insert_one(fw_spec)

@explicit_serialize
class ConsolidateRuns(FireTaskBase):
    _fw_name = "ConsolidateRuns"

    def run_task(self, fw_spec):
        build_doc = copy.deepcopy(fw_spec)
        builds = DB.builds
        pipes = DB.pipes
        build_hash = fw_spec["build_hash"]

        tags_all = []
        template_subdict = {"all": [], "mean": None, "std": None, "max": None,
                            "min": None}
        performance_dict = {k: copy.deepcopy(template_subdict) for k in
                            DATASET_SET}
        features_dict = {k: copy.deepcopy(template_subdict) for k in
                         DATASET_SET}
        time_dict = {k: copy.deepcopy(template_subdict) for k in DATASET_SET}
        samples_dict = {k: copy.deepcopy(template_subdict) for k in DATASET_SET}

        for doc in pipes.find({'build_hash': build_hash}):
            dataset = doc["dataset"]
            tags_all.extend(doc["tags"])
            performance_dict[dataset]["all"].append(doc["score"])
            features_dict[dataset]["all"].append(doc["n_features"])
            samples_dict[dataset]["all"].append(
                doc["test_sample_frac_retained"])
            time_dict[dataset]["all"].append(doc["elapsed_time"])

        build_doc["pipe_config"] = pipes.find_one({'build_hash': build_hash})[
            "pipe_config"]
        build_doc["completion_time"] = datetime.datetime.utcnow()

        for d in [performance_dict, features_dict, samples_dict, time_dict]:
            for ds in d.keys():
                build_list = d[ds]["all"]
                d[ds]["mean"] = np.mean(build_list)
                d[ds]["std"] = np.std(build_list)
                d[ds]["max"] = max(build_list)
                d[ds]["min"] = min(build_list)

        performance_means = [ds["mean"] for ds in performance_dict.values()]
        build_doc["performance_mean"] = np.mean(performance_means)
        build_doc["performance_std"] = np.std(performance_means)
        build_doc["performance_max"] = max(performance_means)
        build_doc["performance_min"] = min(performance_means)

        features_means = [ds["mean"] for ds in features_dict.values()]
        build_doc["features_mean"] = np.mean(features_means)
        build_doc["features_std"] = np.std(features_means)
        build_doc["features_max"] = max(features_means)
        build_doc["features_min"] = min(features_means)

        samples_means = [ds["mean"] for ds in samples_dict.values()]
        build_doc["samples_mean"] = np.mean(samples_means)
        build_doc["samples_std"] = np.std(samples_means)
        build_doc["samples_max"] = max(samples_means)
        build_doc["samples_min"] = min(samples_means)

        time_means = [ds["mean"] for ds in time_dict.values()]
        build_doc["time_mean"] = np.mean(time_means)
        build_doc["time_std"] = np.std(time_means)
        build_doc["time_max"] = max(time_means)
        build_doc["time_min"] = min(time_means)
        builds.insert_one(build_doc)


def get_workflow_from_build(name, pipe_config, base_save_dir, tags=None):
    top_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../..")
    repo = git.Repo(top_dir)
    last_commit = str(repo.head.commit)
    # Build hash is the combination of pipe configuration and current commit
    build_config_for_hash = copy.deepcopy(pipe_config)
    build_config_for_hash["last_commit"] = last_commit
    build_config_for_hash = str(build_config_for_hash).encode("UTF-8")
    build_hash = hashlib.sha1(build_config_for_hash).hexdigest()[:10]
    save_dir = os.path.join(base_save_dir,
                            str(datetime.datetime.utcnow()).replace(" ", "_") + "_" + build_hash)

    pipe_fws = []
    for dataset in DATASET_SET:
        spec = {"dataset": dataset,
                "pipe_config": pipe_config,
                "commit": last_commit,
                "name": name,
                "build_hash": build_hash,
                "tags": tags if tags else [],
                "save_dir": os.path.join(save_dir, dataset)}
        for trial in range(N_TRIALS):
            spec["trial"] = trial
            pipe_fws.append(Firework([RunPipe(), StorePipeResults()],
                                     spec=spec,
                                     name="{}: {} - trial {}".format(name,
                                                                     dataset,
                                                                     trial)))
    fw_consolidate = Firework(ConsolidateRuns(),
                              spec={"build_hash": build_hash},
                              name="Consolidate build {}".format(build_hash))

    links = {fw: [fw_consolidate] for fw in pipe_fws}
    wf = Workflow(pipe_fws + [fw_consolidate],
                  links_dict=links,
                  name="{}: build {}".format(name, build_hash))
    return wf


if __name__ == "__main__":
    # pipe_config = {"learner_name": "TPOTAdaptor",
    #                "learner_kwargs": {"max_time_mins": 2,
    #                                   "population_size": 10},
    #                "reducer_kwargs": {},
    #                "autofeaturizer_kwargs": {},
    #                "cleaner_kwargs": {}}
    #
    # wf = get_workflow_from_build("TestName", pipe_config, "/Users/ardunn/MSLEARNTEST/")
    #
    lp = LaunchPad(host="localhost", port=27017, name="automatminer")
    # lp.reset(password=None, require_password=False)
    # lp.add_wf(wf)
    launch_rocket(lp)




