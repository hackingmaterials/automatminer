"""
Utils specific to this package.
"""
import json
import os
from pprint import pformat

import pandas as pd
import yaml
from sklearn.exceptions import NotFittedError
from sklearn.pipeline import Pipeline

AMM_SUPPORTED_EXTS = [".txt", ".json", ".yaml", ".yml", ""]


class AutomatminerError(BaseException):
    """
    Exception specific to automatminer methods.
    """

    def __init__(self, msg):
        self.msg = msg

    def __str__(self):
        return "AutomatminerError : " + self.msg


class VersionError(AutomatminerError):
    """
    Version errors
    """

    def __str__(self):
        base_str = super(VersionError, self).__str__()
        return base_str + " (VersionError)"


def compare_columns(df1, df2, ignore=None) -> dict:
    """
    Compare the columns of a dataframe.

    Args:
        df1 (pandas.DataFrame): The first dataframe.
        df2 (pandas.DataFrame): The second dataframe.
        ignore ([str]): The feature labels to ignore in the analyis.

    Returns:
        (dict): {"df1_not_in_df2": [The columns in df1 not in df2],
                 "df2_not_in_df1": [The columns in df2 not in df1],
                 "mismatch": (bool)}
    """
    ignore = () if ignore is None else ignore
    df2_not_in_df1 = [
        f for f in df2.columns if f not in df1.columns and f not in ignore
    ]
    df1_not_in_df2 = [
        f for f in df1.columns if f not in df2.columns and f not in ignore
    ]
    matched = not (df2_not_in_df1 + df1_not_in_df2)
    return {
        "df2_not_in_df1": df2_not_in_df1,
        "df1_not_in_df2": df1_not_in_df2,
        "mismatch": not matched,
    }


def check_fitted(func):
    """
    Decorator to check if a transformer has been fitted.
    Args:
        func: A function or method.

    Returns:
        A wrapper function for the input function/method.
    """

    def wrapper(*args, **kwargs):
        if not hasattr(args[0], "is_fit"):
            raise AttributeError(
                "Method using check_fitted has no is_fit attr" " to check if fitted!"
            )
        if not args[0].is_fit:
            raise NotFittedError(
                "{} has not been fit!" "".format(args[0].__class__.__name__)
            )
        else:
            return func(*args, **kwargs)

    return wrapper


def set_fitted(func):
    """
    Decorator to ensure a transformer is fitted properly.
    Args:
        func: A function or method.

    Returns:
        A wrapper function for the input function/method.
    """

    def wrapper(*args, **kwargs):
        args[0].is_fit = False
        result = func(*args, **kwargs)
        args[0].is_fit = True
        return result

    return wrapper


def return_attrs_recursively(obj) -> dict:
    """
    Returns attributes of an object recursively. Stops recursion when
    attrs go outside of the automatminer library.

    Args:
        obj (object): The object with attrs

    Returns:
        attrdict (dict): The dictionary containing attributes which can
            be pretty-printed.
    """
    attrdict = {}
    for attr, value in obj.__dict__.items():
        if hasattr(value, "__dict__") and hasattr(value, "__module__"):
            if "automatminer" in value.__module__:
                attrdict[attr] = {attr: return_attrs_recursively(value)}
            elif isinstance(value, pd.DataFrame):
                attrdict[attr] = {
                    "obj": value.__class__,
                    "columns": value.shape[1],
                    "samples": value.shape[0],
                }
            elif isinstance(value, Pipeline):
                attrdict[attr] = [str(s) for s in value.steps]
            else:
                attrdict[attr] = value
        else:
            # Prevent huge matrices being spammed to the digest
            if "ml_data" not in attr:
                attrdict[attr] = value
    return attrdict


def save_dict_to_file(d, filename) -> None:
    """
    Save a dictionary to a persistent file. Supported formats and extensions are
    text ('.txt'), JSON ('.json'), and YAML ('.yaml', '.yml').

    If no extension is provided, text format will be used.

    Args:
        d (dict): A dictionary of strings or objects castable to python native
            objects (e.g., NumPy integers).
        filename (str): The filename and extension to save the file. For
            example, "mydict.json".

    Returns:
        None
    """
    fname, ext = os.path.splitext(filename)

    if ext in (".json", ".yaml", ".yml"):
        digest = json.dumps(d, default=lambda x: str(x))
        if ext in (".yaml", ".yml"):
            digest = yaml.dump(yaml.safe_load(digest))
    elif ext in (".txt", "", None):
        digest = pformat(d)
    else:
        raise ValueError(
            f"The extension {ext} in filename {fname} is no supported. Use a "
            f"supported extension: {AMM_SUPPORTED_EXTS}"
        )

    with open(filename, "w") as f:
        f.write(digest)


def get_version():
    """
    Get the version of automatminer without worrying about circular imports in
    __init__.

    Returns:
        (str): the version

    """
    thisdir = os.path.dirname(os.path.realpath(__file__))
    version_reference = os.path.join(thisdir, "../__init__.py")
    with open(version_reference, "r") as f:
        init_file = f.readlines()
        v = [v for v in init_file if "__version__" in v][0]
    v = v.replace("__version__", "").replace('"', "").replace("=", "").strip()
    return v
