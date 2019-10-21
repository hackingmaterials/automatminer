import os
import datetime

from fireworks import Firework, ScriptTask
import git
import automatminer

from automatminer_dev.config import RUN_TESTS_CMD, EXPORT_COV_CMD

VALID_FWORKERS = ["local", "cori", "lrc"]


def get_last_commit():
    file = automatminer.__file__
    top_dir = os.path.join(os.path.dirname(file), "../")
    repo = git.Repo(top_dir)
    return str(repo.head.commit)


def get_time_str():
    return datetime.datetime.now().strftime("%Y.%m.%d_%H:%M:%S")


def get_test_fw(fworker, build_id=None, add_to_spec=None):
    spec = {"_fworker": fworker}

    if not build_id:
        build_id = "no_build"

    if add_to_spec:
        spec.update(add_to_spec)

    run_test = ScriptTask(script=RUN_TESTS_CMD)
    export_coverage = ScriptTask(script=EXPORT_COV_CMD)
    fw_test = Firework(
        [run_test, export_coverage],
        spec=spec,
        name="run tests ({})".format(build_id)
    )
    return fw_test
