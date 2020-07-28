"""Deployment file to facilitate releases.
"""
import os
import json
import webbrowser
import datetime
import requests
from invoke import task
from automatminer import __version__
from monty.os import cd

__author__ = ["Alex Dunn", "Shyue Ping Ong", "Anubhav Jain"]


# Making and updatig documentation
@task
def make_doc(ctx):
    with cd("docs"):
        ctx.run("sphinx-apidoc -o ./source -f ../automatminer")
        ctx.run("make html")
        # ctx.run("cp _static/* ../docs/html/_static")
        ctx.run("cp -r build/html/* .")
        ctx.run("rm -r build")
        ctx.run("touch .nojekyll")


@task
def open_doc(ctx):
    pth = os.path.abspath("docs/index.html")
    webbrowser.open("file://" + pth)


@task
def version_check(ctx):
    with open("setup.py", "r") as f:
        setup_version = None
        for l in f.readlines():
            if "version = " in l:
                setup_version = l.split(" ")[-1]
                setup_version = setup_version.replace('"', "").replace("\n", "")

    if setup_version is None:
        raise IOError("Could not parse setup.py for version.")

    if __version__ == setup_version:
        print("Setup and init versions match eachother.")
        today = datetime.date.today().strftime("%Y%m%d")
        if today not in __version__:
            raise ValueError(f"The version {__version__} does not match "
                             f"the date format {today}!")
        else:
            print("Date is contained within the version.")
    else:
        raise ValueError(f"There is a mismatch in the date between the "
                         f"rocketsled __init__ and the setup. Please "
                         f"make sure they are the same."
                         f"\n DIFF: {__version__}, {setup_version}")


@task
def update_changelog(ctx):
    ctx.run('github_changelog_generator --user hackingmaterials --project automatminer')
    ctx.run("git add CHANGELOG.md")
    ctx.run("git commit CHANGELOG.md -m 'update changelog [skip ci]'")


@task
def full_tests_circleci(ctx):
    ctx.run("./dev_scripts/run_intensive.sh")


@task
def release(ctx):
    version_check(ctx)
    payload = {
        "tag_name": "v" + __version__,
        "target_commitish": "master",
        "name": "v" + __version__,
        "body": "",
        "draft": False,
        "prerelease": False
    }
    response = requests.post(
        "https://api.github.com/repos/hackingmaterials/automatminer/releases",
        data=json.dumps(payload),
        headers={
            "Authorization": "token " + os.environ["GITHUB_RELEASES_TOKEN"]})
    print(response.text)


@task
def publish(ctx):
    version_check(ctx)
    ctx.run("rm -r dist build", warn=True)
    ctx.run("python3 setup.py sdist bdist_wheel")
    ctx.run("twine upload dist/* --verbose")
