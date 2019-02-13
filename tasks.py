"""Deployment file to facilitate releases.
"""
import os
import json
import webbrowser
import requests
from invoke import task
from automatminer import __version__
from monty.os import cd


__author__ = ["Alex Dunn", "Shyue Ping Ong", "Anubhav Jain"]


@task
def make_doc(ctx):
    with cd("docs"):
        ctx.run("sphinx-apidoc -o . -f .")
        ctx.run("make html")
        # ctx.run("cp _static/* ../docs/html/_static")
        ctx.run("cp -r build/html/* .")
        ctx.run("rm -r build")
        ctx.run("touch .nojekyll")

@task
def publish(ctx):
    ctx.run("rm dist/*.*", warn=True)
    ctx.run("python3 setup.py sdist bdist_wheel")
    ctx.run("twine upload dist/* --verbose")


@task
def update_changelog(ctx):
    ctx.run('github_changelog_generator hackingmaterials/automatminer')


@task
def full_tests_circleci(ctx):
    ctx.run("./all_test_ci.sh")

@task
def release(ctx):
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
        headers={"Authorization": "token " + os.environ["GITHUB_RELEASES_TOKEN"]})
    print(response.text)

#TODO: HAVE THIS PUSH A NEW BENCHMARK TO LAWRENCIUM

@task
def open_doc(ctx):
    pth = os.path.abspath("docs/index.html")
    webbrowser.open("file://" + pth)