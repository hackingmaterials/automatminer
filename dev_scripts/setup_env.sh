echo "SKIPPING INTENSIVE TESTS? $SKIP_INTENSIVE"
python3 -m venv test_env
. test_env/bin/activate
pip install -q --upgrade pip
pip install --quiet -r requirements.txt

# temporarily avoid pip install of matminer
mkdir matminer
cd matminer
git clone https://github.com/hackingmaterials/matminer.git
cd matminer
pip install -q -e .
cd ../..

pip install -e .
pip install -q coverage
pip install -q codacy-coverage