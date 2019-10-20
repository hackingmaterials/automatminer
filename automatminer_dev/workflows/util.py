import os
import datetime

import git
import automatminer

VALID_FWORKERS = ["local", "cori", "lrc"]


def get_last_commit():
    file = automatminer.__file__
    top_dir = os.path.join(os.path.dirname(file), "../")
    repo = git.Repo(top_dir)
    return str(repo.head.commit)


def get_time_str():
    return datetime.datetime.now().strftime("%Y.%m.%d_%H:%M:%S")
