import os
import hashlib
import copy
import random

from fireworks import Firework, Workflow
from sklearn.model_selection import KFold

from automatminer_dev.tasks.bench import (
    ConsolidatePipesToBenchmark,
    RunPipe,
    StorePipeResults,
    ConsolidateBenchmarksToBuild,
)
from automatminer_dev.workflows.util import (
    get_last_commit,
    get_time_str,
    VALID_FWORKERS,
)

from automatminer_dev.config import LP, KFOLD_DEFAULT
from automatminer_dev.workflows.util import get_test_fw

"""
Functions for creating benchmarking workflows.

A pipe is one config being run on one fold of one problem.

A benchmark is one config being run on all nested CV folds of one problem.

A build is getting the results of one config on all problems.


Build
|
N Benchmark(s)
|
M * N Pipe(s)
"""


def wf_evaluate_build(
    fworker,
    build_name,
    dataset_set,
    pipe_config,
    include_tests=False,
    cache=True,
    kfold_config=KFOLD_DEFAULT,
    tags=None,
):
    """
    Current fworkers:
    - "local": Alex's local computer
    - "cori": Cori
    - "lrc": Lawrencium
    """
    if fworker not in VALID_FWORKERS:
        raise ValueError("fworker must be in {}".format(VALID_FWORKERS))

    # Get a fun unique id for this build
    word_file = "/usr/share/dict/words"
    words = open(word_file).read().splitlines()
    words_short = [w for w in words if 4 <= len(w) <= 6]

    build_id = None
    while (
        LP.db.automatminer_builds.find({"build_id": build_id}).count() != 0
        or not build_id
    ):
        build_id = " ".join([w.lower() for w in random.sample(words_short, 2)])
    print("build id: {}".format(build_id))

    all_links = {}
    fws_fold0 = []
    fws_consolidate = []
    benchmark_hashes = []
    for benchmark in dataset_set:
        links, fw_fold0, fw_consolidate = wf_benchmark(
            fworker,
            pipe_config,
            **benchmark,
            tags=tags,
            kfold_config=kfold_config,
            cache=cache,
            return_fireworks=True,
            build_id=build_id,
            add_dataset_to_names=True
        )
        all_links.update(links)
        fws_fold0.extend(fw_fold0)
        fws_consolidate.append(fw_consolidate)
        # benchmark has is the same between all fws in one benchmark
        benchmark_hashes.append(fw_fold0[0].to_dict()["spec"]["benchmark_hash"])

    fw_build_merge = Firework(
        ConsolidateBenchmarksToBuild(),
        spec={
            "benchmark_hashes": benchmark_hashes,
            "build_id": build_id,
            "pipe_config": pipe_config,
            "build_name": build_name,
            "commit": get_last_commit(),
            "_fworker": fworker,
            "tags": tags,
        },
        name="build merge ({})".format(build_id),
    )

    for fw in fws_consolidate:
        all_links[fw] = [fw_build_merge]

    if include_tests:
        fw_test = get_test_fw(fworker, build_id)
        all_links[fw_test] = fws_fold0
    all_links[fw_build_merge] = []

    wf_name = "build: {} ({}) [{}]".format(build_id, build_name, fworker)
    wf = Workflow(
        list(all_links.keys()),
        all_links,
        name=wf_name,
        metadata={
            "build_id": build_id,
            "tags": tags,
            "benchmark_hashes": benchmark_hashes,
        },
    )
    return wf


def wf_benchmark(
    fworker,
    pipe_config,
    name,
    data_file,
    target,
    problem_type,
    clf_pos_label,
    cache=True,
    kfold_config=KFOLD_DEFAULT,
    tags=None,
    return_fireworks=False,
    add_dataset_to_names=True,
    build_id=None,
    prepend_name="",
):
    if fworker not in VALID_FWORKERS:
        raise ValueError("fworker must be in {}".format(VALID_FWORKERS))

    # if fworker == "cori":
    #     n_cori_jobs = 32
    #     warnings.warn(
    #         "Worker is cori. Overriding n_jobs to {}".format(n_cori_jobs))
    #     pipe_config["learner_kwargs"]["n_jobs"] = n_cori_jobs
    #     pipe_config["autofeaturizer_kwargs"]["n_jobs"] = n_cori_jobs

    # Single (run) hash is the combination of pipe configuration + last commit
    # + data_file
    last_commit = get_last_commit()
    benchmark_config_for_hash = copy.deepcopy(pipe_config)
    benchmark_config_for_hash["last_commit"] = last_commit
    benchmark_config_for_hash["data_file"] = data_file
    benchmark_config_for_hash["worker"] = fworker
    benchmark_config_for_hash = str(benchmark_config_for_hash).encode("UTF-8")
    benchmark_hash = hashlib.sha1(benchmark_config_for_hash).hexdigest()[:10]
    base_save_dir = get_time_str() + "_" + benchmark_hash

    common_spec = {
        "pipe_config": pipe_config,
        "base_save_dir": base_save_dir,
        "kfold_config": kfold_config,
        "data_file": data_file,
        "target": target,
        "clf_pos_label": clf_pos_label,
        "problem_type": problem_type,
        "automatminer_commit": last_commit,
        "name": name,
        "benchmark_hash": benchmark_hash,
        "tags": tags if tags else [],
        "cache": cache,
        "build_id": build_id,
        "_fworker": fworker,
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
            pipename = "{}fold {} + featurization ({})".format(
                dataset_name, fold, benchmark_hash
            )
        else:
            pipename = "{}fold {} ({})".format(dataset_name, fold, benchmark_hash)

        fws_all_folds.append(
            Firework([RunPipe(), StorePipeResults()], spec=foldspec, name=pipename)
        )

    fw_consolidate = Firework(
        ConsolidatePipesToBenchmark(),
        spec=common_spec,
        name="bench merge ({})".format(benchmark_hash),
    )

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
        wf_name = "benchmark {}: ({}) [{}]".format(benchmark_hash, name, fworker)
        if prepend_name:
            wf_name = "<<{}>> {}".format(prepend_name, wf_name)

        wf = Workflow(
            list(links.keys()),
            links_dict=links,
            name=wf_name,
            metadata={"benchmark_hash": benchmark_hash, "tags": tags},
        )
        return wf
