echo "SKIPPING INTENSIVE TESTS? $SKIP_INTENSIVE"
python3 -m venv test_env
. test_env/bin/activate
pip install -q --upgrade pip

pip install -e .
pip install -q coverage
pip install -q codacy-coverage