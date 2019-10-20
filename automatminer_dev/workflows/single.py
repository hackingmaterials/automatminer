import warnings

from fireworks import Firework, Workflow

from automatminer_dev.tasks.single import RunSingleFit
from automatminer_dev.workflows.util import get_last_commit, get_time_str, \
    VALID_FWORKERS


def wf_single_fit(fworker, fit_name, pipe_config, name, data_file, target,
                  *args,
                  tags=None, **kwargs):
    """
    Submit a dataset to be fit for a single pipeline (i.e., to train on a
    dataset for real predictions).
    """
    warnings.warn("Single fitted MatPipe not being stored in automatminer db "
                  "collections. Please consult fw_spec to find the benchmark "
                  "on {}".format(fworker))
    if fworker not in VALID_FWORKERS:
        raise ValueError("fworker must be in {}".format(VALID_FWORKERS))

    base_save_dir = get_time_str() + "_single_fit"

    spec = {
        "pipe_config": pipe_config,
        "base_save_dir": base_save_dir,
        "data_file": data_file,
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
