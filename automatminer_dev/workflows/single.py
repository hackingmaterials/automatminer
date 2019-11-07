import os
import warnings
import subprocess
from datetime import datetime

from fireworks import Firework, Workflow
from paramiko import SSHClient
from scp import SCPClient
from matminer.utils.io import store_dataframe_as_json

from automatminer_dev.tasks.single import RunSingleFit
from automatminer_dev.workflows.util import get_test_fw, get_last_commit
from automatminer_dev.workflows.util import (
    get_last_commit,
    get_time_str,
    VALID_FWORKERS,
)


def wf_single_fit(
        fworker, fit_name, pipe_config, name, df, target, tags=None
):
    """
    Submit a dataset to be fit for a single pipeline (i.e., to train on a
    dataset for real predictions).
    """

    # todo this is not working probably
    warnings.warn(
        "Single fitted MatPipe not being stored in automatminer db "
        "collections. Please consult fw_spec to find the benchmark "
        "on {}".format(fworker)
    )
    if fworker not in VALID_FWORKERS:
        raise ValueError("fworker must be in {}".format(VALID_FWORKERS))

    data_file = None

    now = get_time_str()
    base_save_dir = now + "_single_fit"

    spec = {
        "pipe_config": pipe_config,
        "base_save_dir": base_save_dir,
        "data_file": data_file,
        "target": target,
        "automatminer_commit": get_last_commit(),
        "tags": tags if tags else [],
        "_fworker": fworker,
    }

    fw_name = "{} single fit".format(name)
    wf_name = "single fit: {} ({}) [{}]".format(name, fit_name, fworker)

    fw = Firework(RunSingleFit(), spec=spec, name=fw_name)
    wf = Workflow([fw], metadata={"tags": tags}, name=wf_name)
    return wf


def wf_run_test(fworker, test_name):
    commit = get_last_commit()
    wf_name = "run tests: {} [{}]".format(test_name, commit)

    add_to_spec = {
        "commit": commit,
    }

    test_fw = get_test_fw(fworker, add_to_spec=add_to_spec)
    wf = Workflow([test_fw], metadata={"tags": "test"}, name=wf_name)
    return wf


# This does work
def transfer_data(df, worker, now):
    this_dir = os.path.dirname(os.path.abspath(__file__))
    user_folder = os.path.join(this_dir, "user_dfs")
    if not os.path.exists(user_folder):
        os.makedirs(user_folder)
    filename = "user_df_" + now + ".json"
    filepath = os.path.join(user_folder, filename)
    store_dataframe_as_json(df, filepath)

    if worker != "local":
        if worker == "cori":
            o = subprocess.check_output(
                ['bash', '-c', '. ~/.bash_profile; cori_get_password']
            )
            user = os.environ["CORI_USER"]
            host = "lrc-login.lbl.gov"
        elif worker == "lrc":
            o = subprocess.check_output(
                ['bash', '-c', '. ~/.bash_profile; lrc_get_password']
            )
            user = os.environ["LRC_USER"]
            host = "lrc-login.lbl.gov"
        else:
            raise ValueError(f"Worker {worker} not valid!")

        o_utf = o.decode("utf-8")
        o_all = o_utf.split("\n")
        o_all.remove("")
        password = o_all[-1]

        ssh = SSHClient()
        ssh.load_system_host_keys()
        ssh.connect(host, username=user, password=password, look_for_keys=False)

        with SCPClient(ssh.get_transport()) as scp:
            scp.put(filepath, recursive=True,
                    remote_path="/global/home/users/ardunn")
    else:
        pass


if __name__ == "__main__":
    import pandas as pd
    from matminer.datasets import load_dataset
    from automatminer_dev.workflows.util import get_time_str

    df = load_dataset("matbench_jdft2d")
    transfer_data(df, "lrc", get_time_str())
