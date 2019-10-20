import os
import time
import copy
import datetime
import warnings
from math import sqrt

import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold
from fireworks import FireTaskBase, explicit_serialize
from sklearn.metrics import (
    f1_score,
    r2_score,
    mean_squared_error,
    mean_absolute_error,
    roc_auc_score,
    accuracy_score,
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from matminer.utils.io import load_dataframe_from_json

from automatminer.featurization import AutoFeaturizer
from automatminer.preprocessing import DataCleaner, FeatureReducer
from automatminer.automl.adaptors import TPOTAdaptor, SinglePipelineAdaptor
from automatminer.pipeline import MatPipe
from automatminer.utils.ml import AMM_REG_NAME, AMM_CLF_NAME

from automatminer_dev.config import LP

"""
Tasks for dev benchmarking.

"""


@explicit_serialize
class RunPipe(FireTaskBase):
    _fw_name = "RunPipe"

    def run_task(self, fw_spec):
        # Read data from fw_spec
        pipe_config_dict = fw_spec["pipe_config"]
        fold = fw_spec["fold"]
        kfold_config = fw_spec["kfold_config"]
        target = fw_spec["target"]
        data_file = fw_spec["data_file"]
        clf_pos_label = fw_spec["clf_pos_label"]
        problem_type = fw_spec["problem_type"]
        learner_name = pipe_config_dict["learner_name"]
        cache = fw_spec["cache"]
        learner_kwargs = pipe_config_dict["learner_kwargs"]
        reducer_kwargs = pipe_config_dict["reducer_kwargs"]
        cleaner_kwargs = pipe_config_dict["cleaner_kwargs"]
        autofeaturizer_kwargs = pipe_config_dict["autofeaturizer_kwargs"]

        # Modify data_file based on computing resource
        data_dir = os.environ["AMM_DATASET_DIR"]
        data_file = os.path.join(data_dir, data_file)

        # Modify save_dir based on computing resource
        bench_dir = os.environ["AMM_BENCH_DIR"]
        base_save_dir = fw_spec["base_save_dir"]
        base_save_dir = os.path.join(bench_dir, base_save_dir)
        save_dir = fw_spec.pop("save_dir")
        save_dir = os.path.join(base_save_dir, save_dir)

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Set up pipeline config
        if learner_name == "TPOTAdaptor":
            learner = TPOTAdaptor(**learner_kwargs)
        elif learner_name == "rf":
            warnings.warn(
                "Learner kwargs passed into RF regressor/classifiers bc. rf being used."
            )
            learner = SinglePipelineAdaptor(
                regressor=RandomForestRegressor(**learner_kwargs),
                classifier=RandomForestClassifier(**learner_kwargs),
            )
        else:
            raise ValueError("{} not supported by RunPipe yet!" "".format(learner_name))
        if cache:
            autofeaturizer_kwargs["cache_src"] = os.path.join(
                base_save_dir, "features.json"
            )
        pipe_config = {
            "learner": learner,
            "reducer": FeatureReducer(**reducer_kwargs),
            "cleaner": DataCleaner(**cleaner_kwargs),
            "autofeaturizer": AutoFeaturizer(**autofeaturizer_kwargs),
        }

        pipe = MatPipe(**pipe_config)

        # Set up dataset
        # Dataset should already be set up correctly as json beforehand.
        # this includes targets being converted to classification, removing
        # extra columns, having the names of featurization cols set to the
        # same as the matpipe config, etc.
        df = load_dataframe_from_json(data_file)

        # Check other parameters that would otherwise not be checked until after
        # benchmarking, hopefully saves some errors at the end during scoring.
        if problem_type not in [AMM_CLF_NAME, AMM_REG_NAME]:
            raise ValueError("Problem must be either classification or " "regression.")
        elif problem_type == AMM_CLF_NAME:
            if not isinstance(clf_pos_label, (str, bool)):
                raise TypeError(
                    "The classification positive label should be a "
                    "string, or bool not {}."
                    "".format(type(clf_pos_label))
                )
            elif clf_pos_label not in df[target]:
                raise ValueError(
                    "The classification positive label should be"
                    "present in the target column."
                )
            elif len(df[target].unique()) > 2:
                raise ValueError(
                    "Only binary classification scoring available" "at this time."
                )

        # Set up testing scheme
        if problem_type == AMM_REG_NAME:
            kfold = KFold(**kfold_config)
        else:
            kfold = StratifiedKFold(**kfold_config)
        if fold >= kfold.n_splits:
            raise ValueError(
                "{} is out of range for KFold with n_splits=" "{}".format(fold, kfold)
            )

        # Run the benchmark
        t1 = time.time()
        results = pipe.benchmark(df, target, kfold, fold_subset=[fold], cache=True)
        result_df = results[0]
        elapsed_time = time.time() - t1

        # Save everything
        pipe.save(os.path.join(save_dir, "pipe.p"))
        pipe.inspect(filename=os.path.join(save_dir, "digest.txt"))
        result_df.to_csv(os.path.join(save_dir, "test_df.csv"))
        pipe.post_fit_df.to_csv(os.path.join(save_dir, "fitted_df.csv"))

        # Evaluate model
        true = result_df[target]
        test = result_df[target + " predicted"]

        pass_to_storage = {}
        if problem_type == AMM_REG_NAME:
            pass_to_storage["r2"] = r2_score(true, test)
            pass_to_storage["mae"] = mean_absolute_error(true, test)
            pass_to_storage["rmse"] = sqrt(mean_squared_error(true, test))
        elif problem_type == AMM_CLF_NAME:
            pass_to_storage["f1"] = f1_score(true, test, pos_label=clf_pos_label)
            pass_to_storage["roc_auc"] = roc_auc_score(true, test)
            pass_to_storage["accuracy"] = accuracy_score(true, test)
        else:
            raise ValueError(
                "Scoring method for problem type {} not supported"
                "".format(problem_type)
            )

        # Extract important inspect for storage
        try:
            # TPOT Adaptor
            best_pipeline = [str(step) for step in pipe.learner.best_pipeline.steps]
        except AttributeError:
            best_pipeline = str(pipe.learner.best_pipeline)

        features = pipe.learner.features
        n_features = len(features)
        fold_orig = list(kfold.split(df, y=df[target]))[fold]
        n_samples_train_original = len(fold_orig[0])
        n_samples_test_original = len(fold_orig[1])

        pass_to_storage.update(
            {
                "target": target,
                "best_pipeline": best_pipeline,
                "elapsed_time": elapsed_time,
                "features": features,
                "n_features": n_features,
                "n_test_samples_original": n_samples_test_original,
                "n_train_samples_original": n_samples_train_original,
                "n_train_samples": len(pipe.post_fit_df),
                "n_test_samples": len(test),
                "test_sample_frac_retained": len(test) / n_samples_test_original,
                "completion_time": datetime.datetime.now(),
                "base_save_dir": base_save_dir,
                "save_dir": save_dir,
            }
        )
        fw_spec.update(pass_to_storage)


@explicit_serialize
class StorePipeResults(FireTaskBase):
    """
    Store the results for a single pipeline. One fold ran on one dataset
    under one configuration. May be within a build.
    """

    _fw_name = "StorePipeResults"

    def run_task(self, fw_spec):
        storable = copy.deepcopy(fw_spec)
        storable.pop("_fw_env", None)
        storable.pop("_tasks", None)
        LP.db.automatminer_pipes.insert_one(storable)


@explicit_serialize
class ConsolidatePipesToBenchmark(FireTaskBase):
    """
    Consolidate pipelines into a doc for one pipe_config on one dataset.

    Benchmarks are identified by their benchmark_hash (e.g. 4207c52963)
    """

    _fw_name = "ConsolidatePipesToBenchmark"

    def run_task(self, fw_spec):
        benchmark_doc = copy.deepcopy(fw_spec)
        benchmark_doc.pop("_tasks", None)
        benchmark_doc.pop("_fw_env", None)
        benchmarks = LP.db.automatminer_benchmarks
        pipes = LP.db.automatminer_pipes

        benchmark_hash = fw_spec["benchmark_hash"]
        benchmark_doc["dataset_name"] = fw_spec["name"]
        benchmark_doc.pop("name", None)

        template_subdict = {
            "all": {},
            "mean": None,
            "std": None,
            "max": None,
            "min": None,
        }
        problem_type = fw_spec["problem_type"]
        if problem_type == AMM_REG_NAME:
            scores_dict = {
                "rmse": copy.deepcopy(template_subdict),
                "mae": copy.deepcopy(template_subdict),
                "r2": copy.deepcopy(template_subdict),
            }
        elif problem_type == AMM_CLF_NAME:
            scores_dict = {
                "f1": copy.deepcopy(template_subdict),
                "roc_auc": copy.deepcopy(template_subdict),
                "accuracy": copy.deepcopy(template_subdict),
            }
        else:
            raise ValueError("problem_type {} not recognized!".format(problem_type))

        n_features_dict = copy.deepcopy(template_subdict)
        time_dict = copy.deepcopy(template_subdict)
        samples_dict = copy.deepcopy(template_subdict)

        for doc in pipes.find({"benchmark_hash": benchmark_hash}):
            fold = "fold_{}".format(doc["fold"])
            n_features_dict["all"][fold] = doc["n_features"]
            samples_dict["all"][fold] = doc["test_sample_frac_retained"]
            time_dict["all"][fold] = doc["elapsed_time"]
            for metric in scores_dict.keys():
                scores_dict[metric]["all"][fold] = doc[metric]

        benchmark_doc["completion_time"] = datetime.datetime.now()

        for d in [n_features_dict, samples_dict, time_dict]:
            benchmark_list = list(d["all"].values())
            d["mean"] = np.mean(benchmark_list)
            d["std"] = np.std(benchmark_list)
            d["max"] = max(benchmark_list)
            d["min"] = min(benchmark_list)

        for metric in scores_dict.keys():
            benchmark_list = list(scores_dict[metric]["all"].values())
            scores_dict[metric]["mean"] = np.mean(benchmark_list)
            scores_dict[metric]["std"] = np.std(benchmark_list)
            scores_dict[metric]["max"] = max(benchmark_list)
            scores_dict[metric]["min"] = min(benchmark_list)

        benchmark_doc.update(scores_dict)
        benchmark_doc["n_features"] = n_features_dict
        benchmark_doc["test_sample_frac_retained"] = samples_dict
        benchmark_doc["elapsed_time"] = time_dict
        benchmarks.insert_one(benchmark_doc)


@explicit_serialize
class ConsolidateBenchmarksToBuild(FireTaskBase):
    """
    Consolidate benchmarks (one pipe_config on one dataset) to a build (one
    pipe_config on a set of datasets)

    Builds are identified uniquely by their build_id (e.g., 'Warp Fen Kras').
    """

    _fw_name = "ConsolidateBenchmarksToBuild"

    def run_task(self, fw_spec):
        benchmark_hashes = fw_spec["benchmark_hashes"]
        benchmarks = LP.db.automatminer_benchmarks
        builds = LP.db.automatminer_builds
        build_doc = copy.deepcopy(fw_spec)
        build_doc.pop("_tasks", None)
        build_doc.pop("_fw_env", None)
        build_id = fw_spec["build_id"]

        for bhash in benchmark_hashes:
            n_bdocs = benchmarks.find(
                {"benchmark_hash": bhash, "build_id": build_id}
            ).count()
            if n_bdocs != 1:
                raise ValueError(
                    "The number of benchmark docs with this build is {}, not 1!".format(
                        n_bdocs
                    )
                )
            else:
                bench_doc = benchmarks.find_one(
                    {"benchmark_hash": bhash, "build_id": build_id}
                )

                if bench_doc["problem_type"] == AMM_REG_NAME:
                    score_types = ["r2", "mae", "rmse"]
                else:
                    score_types = ["f1", "roc_auc", "accuracy"]
                dataset_name = bench_doc["dataset_name"]
                if dataset_name in build_doc:
                    raise ValueError(
                        "{} occurs more than once in the db (hash {})!".format(
                            dataset_name, bhash
                        )
                    )
                build_doc[dataset_name] = {k: bench_doc[k]["mean"] for k in score_types}
                build_doc[dataset_name]["mean_elapsed_time"] = bench_doc[
                    "elapsed_time"
                ]["mean"]
                build_doc[dataset_name]["mean_test_sample_frac_retained"] = bench_doc[
                    "test_sample_frac_retained"
                ]["mean"]
                build_doc[dataset_name]["mean_n_features"] = bench_doc["n_features"][
                    "mean"
                ]
                build_doc[dataset_name]["benchmark_hash"] = bhash
                build_doc[dataset_name]["problem_type"] = bench_doc["problem_type"]
        builds.insert_one(build_doc)
