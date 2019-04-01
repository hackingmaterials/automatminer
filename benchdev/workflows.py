import os
import hashlib
import copy
import datetime
import random
import warnings

import git
import automatminer
from fireworks import Firework, Workflow, ScriptTask, LaunchPad, FWorker
from sklearn.model_selection import KFold

from benchdev.tasks import \
    ConsolidatePipesToBenchmark, RunPipe, StorePipeResults, \
    ConsolidateBenchmarksToBuild, RunSingleFit
from benchdev.config import LP, KFOLD_DEFAULT, \
    LOCAL_DEBUG_SET, RUN_TESTS_CMD, BENCHMARK_DEBUG_SET, BENCHMARK_FULL_SET

"""
Functions for creating benchmarks.

"""

valid_fworkers = ["local", "cori", "lrc"]


def get_last_commit():
    file = automatminer.__file__
    top_dir = os.path.join(os.path.dirname(file), "../")
    repo = git.Repo(top_dir)
    return str(repo.head.commit)

def get_time_str():
    return datetime.datetime.now().strftime("%Y.%m.%d_%H:%M:%S")

def check_pipe_config(pipe_config):
    if "logger" in pipe_config:
        raise ValueError("Logger is set internally by tasks.")


def wf_single_fit(fworker, fit_name, pipe_config, name, data_pickle, target, *args,
                  tags=None, **kwargs):
    """
    Submit a dataset to be fit for a single pipeline (i.e., to train on a
    dataset for real predictions).
    """
    check_pipe_config(pipe_config)
    warnings.warn("Single fitted MatPipe not being stored in automatminer db "
                  "collections. Please consult fw_spec to find the benchmark "
                  "on {}".format(fworker))
    if fworker not in valid_fworkers:
        raise ValueError("fworker must be in {}".format(valid_fworkers))

    base_save_dir = get_time_str() + "_single_fit"

    spec = {
        "pipe_config": pipe_config,
        "base_save_dir": base_save_dir,
        "data_pickle": data_pickle,
        "target": target,
        "automatminer_commit": get_last_commit(),
        "tags": tags if tags else [],
        "_fworker": fworker
    }

    fw_name = "{} single fit".format(name)
    wf_name = "single fit: {} ({}) [{}]".format(name, fit_name, fworker)

    fw = Firework(RunSingleFit(), spec=spec, name=fw_name)
    wf = Workflow([fw], metadata={"tags": tags}, name=wf_name)
    return wf


def wf_evaluate_build(fworker, build_name, dataset_set, pipe_config,
                      include_tests=False, cache=True,
                      kfold_config=KFOLD_DEFAULT, tags=None):
    """
    Current fworkers:
    - "local": Alex's local computer
    - "cori": Cori
    - "lrc": Lawrencium
    """
    check_pipe_config(pipe_config)
    if fworker not in valid_fworkers:
        raise ValueError("fworker must be in {}".format(valid_fworkers))

    # Get a fun unique id for this build
    word_file = "/usr/share/dict/words"
    words = open(word_file).read().splitlines()
    words_short = [w for w in words if 4 <= len(w) <= 6]

    build_id = None
    while LP.db.automatminer_builds.find(
            {"build_id": build_id}).count() != 0 or not build_id:
        build_id = " ".join([w.lower() for w in random.sample(words_short, 2)])
    print("build id: {}".format(build_id))

    all_links = {}
    fws_fold0 = []
    fws_consolidate = []
    benchmark_hashes = []
    for benchmark in dataset_set:
        links, fw_fold0, fw_consolidate = wf_benchmark(fworker, pipe_config,
                                                       **benchmark,
                                                       tags=tags,
                                                       kfold_config=kfold_config,
                                                       cache=cache,
                                                       return_fireworks=True,
                                                       build_id=build_id,
                                                       add_dataset_to_names=True)
        all_links.update(links)
        fws_fold0.extend(fw_fold0)
        fws_consolidate.append(fw_consolidate)
        # benchmark has is the same between all fws in one benchmark
        benchmark_hashes.append(fw_fold0[0].to_dict()["spec"]["benchmark_hash"])

    fw_build_merge = Firework(ConsolidateBenchmarksToBuild(),
                              spec={"benchmark_hashes": benchmark_hashes,
                                    "build_id": build_id,
                                    "pipe_config": pipe_config,
                                    "build_name": build_name,
                                    "commit": get_last_commit(),
                                    "_fworker": fworker,
                                    "tags": tags},
                              name="build merge ({})".format(build_id))

    for fw in fws_consolidate:
        all_links[fw] = [fw_build_merge]

    if include_tests:
        fw_test = Firework(ScriptTask(script=RUN_TESTS_CMD),
                           name="run tests ({})".format(build_id))
        all_links[fw_test] = fws_fold0
    all_links[fw_build_merge] = []

    wf_name = "build: {} ({}) [{}]".format(build_id, build_name, fworker)
    wf = Workflow(list(all_links.keys()), all_links, name=wf_name,
                  metadata={"build_id": build_id, "tags": tags,
                            "benchmark_hashes": benchmark_hashes})
    return wf


def wf_benchmark(fworker, pipe_config, name, data_pickle, target, problem_type,
                 clf_pos_label,
                 cache=True, kfold_config=KFOLD_DEFAULT, tags=None,
                 return_fireworks=False, add_dataset_to_names=True,
                 build_id=None):
    check_pipe_config(pipe_config)
    if fworker not in valid_fworkers:
        raise ValueError("fworker must be in {}".format(valid_fworkers))

    if fworker == "cori":
        n_cori_jobs = 32
        warnings.warn(
            "Worker is cori. Overriding n_jobs to {}".format(n_cori_jobs))
        pipe_config["learner_kwargs"]["n_jobs"] = n_cori_jobs
        pipe_config["autofeaturizer_kwargs"]["n_jobs"] = n_cori_jobs

    # Single (run) hash is the combination of pipe configuration + last commit
    # + data_pickle
    last_commit = get_last_commit()
    benchmark_config_for_hash = copy.deepcopy(pipe_config)
    benchmark_config_for_hash["last_commit"] = last_commit
    benchmark_config_for_hash["data_pickle"] = data_pickle
    benchmark_config_for_hash = str(benchmark_config_for_hash).encode("UTF-8")
    benchmark_hash = hashlib.sha1(benchmark_config_for_hash).hexdigest()[:10]
    base_save_dir = get_time_str() + "_" + benchmark_hash

    common_spec = {
        "pipe_config": pipe_config,
        "base_save_dir": base_save_dir,
        "kfold_config": kfold_config,
        "data_pickle": data_pickle,
        "target": target,
        "clf_pos_label": clf_pos_label,
        "problem_type": problem_type,
        "automatminer_commit": last_commit,
        "name": name,
        "benchmark_hash": benchmark_hash,
        "tags": tags if tags else [],
        "cache": cache,
        "build_id": build_id,
        "_fworker": fworker
    }

    dataset_name = "" if not add_dataset_to_names else name + " "

    fws_all_folds = []
    kfold = KFold(**kfold_config)
    for fold in range(kfold.n_splits):
        save_dir = os.path.join("fold_{}".format(fold))
        foldspec = copy.deepcopy(common_spec)
        foldspec["fold"] = fold
        foldspec["save_dir"] = save_dir

        if fold == 0 and cache:
            pipename = "{}fold {} + featurization ({})".format(dataset_name,
                                                               fold,
                                                               benchmark_hash)
        else:
            pipename = "{}fold {} ({})".format(dataset_name, fold,
                                               benchmark_hash)

        fws_all_folds.append(Firework([RunPipe(), StorePipeResults()],
                                      spec=foldspec,
                                      name=pipename))

    fw_consolidate = Firework(ConsolidatePipesToBenchmark(),
                              spec=common_spec,
                              name="bench merge ({})".format(benchmark_hash))

    if cache:
        fw_fold0 = fws_all_folds[0]
        fws_folds = fws_all_folds[1:]
        links = {fw: [fw_consolidate] for fw in fws_folds}
        links[fw_fold0] = fws_folds
        links[fw_consolidate] = []
    else:
        links = {fw: [fw_consolidate] for fw in fws_all_folds}
        links[fw_consolidate] = []
        fw_fold0 = fws_all_folds

    if return_fireworks:
        connected_to_top_wf = [fw_fold0] if cache else fw_fold0
        return links, connected_to_top_wf, fw_consolidate
    else:
        wf_name = "benchmark {}: ({}) [{}]".format(benchmark_hash, name,
                                                   fworker)
        wf = Workflow(list(links.keys()), links_dict=links,
                      name=wf_name, metadata={"benchmark_hash": benchmark_hash,
                                              "tags": tags}, )
        return wf


if __name__ == "__main__":
    """
    Current tags
    
    Dataset sets
    - data_debug
    - data_full
    
    Feature reduction
    - corr_only
    - pca_only
    - rebate_only
    
    Data cleaning:
    - mean_only
    - drop_mean: drop for features and fit samples, mean for transform
    
    autofeaturizer:
    - af_fast
    - af_best
    
    learner:
    - single_debug
    - single
    - tpot_short
    - tpot_long
    - tpot_generations
    """

    # from tpot.base import TPOTBase
    # TPOTBase(generations=, population_size=)
    pipe_config = {
        "learner_name": "TPOTAdaptor",
        "learner_kwargs": {"generations": 20, "population_size": 20, "max_eval_time_mins": 10},
        # "learner_kwargs": {"max_time_mins": 720, "max_eval_time_mins": 10, "population_size": 100},
        # "learner_kwargs": {"max_time_mins": 360, "population_size": 30},

        # "learner_name": "rf",
        # "learner_kwargs": {"n_estimators": 500},


        "reducer_kwargs": {"reducers": ("corr",)},
        # "reducer_kwargs": {"reducers": ("pca",), "n_pca_features": 0.3},
        # "reducer_kwargs": {"reducers": ("rebate",), "n_rebate_features": 0.3},

        # "reducer_kwargs": {"reducers": ()},
        "autofeaturizer_kwargs": {"preset": "fast"},
        "cleaner_kwargs": {"max_na_frac": 0.01, "feature_na_method": "mean", "na_method_fit": "drop", "na_method_transform": "mean"}
    }


    # from sklearn.ensemble import RandomForestClassifier
    #
    # rf = RandomForestClassifier()
    tags = [
        # "data_full",
        # "corr_only",
        # "drop_mean",
        # "af_best",
        # "tpot_generations",
        "debug"
    ]

    wf = wf_evaluate_build("lrc", "debug 1", BENCHMARK_FULL_SET, pipe_config,
                           include_tests=False, cache=True, tags=tags)
    # wf = wf_evaluate_build("local", "debug", LOCAL_DEBUG_SET,
    #                        pipe_config, include_tests=False, cache=True, tags=tags)


    # ds = LOCAL_DEBUG_SET[0]
    # wf = wf_single_fit("local", "test fit", pipe_config, **ds, tags=["debug"])

    LP.reset(password=None, require_password=False)
    LP.add_wf(wf)
