import os
import datetime

from fireworks import Firework, ScriptTask
import git
import automatminer

from automatminer_dev.config import RUN_TESTS_CMD

VALID_FWORKERS = ["local", "cori", "lrc"]


def get_last_commit():
    file = automatminer.__file__
    top_dir = os.path.join(os.path.dirname(file), "../")
    repo = git.Repo(top_dir)
    return str(repo.head.commit)


def get_time_str():
    return datetime.datetime.now().strftime("%Y.%m.%d_%H:%M:%S")


def get_test_fw(build_id):
    fw_test = Firework(
        ScriptTask(script=RUN_TESTS_CMD), name="run tests ({})".format(build_id)
    )
    return fw_test
