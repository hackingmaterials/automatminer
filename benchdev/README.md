### Automatminer benchmarking dev

`dev` is a collection of dev tools for executing hundred or thousands of machine
learning benchmarks on parallel computing resources with Fireworks. This is not
part of the main automatminer code, and as such, is not
   1. maintained as closely as the main code
   2. tested as rigorously as the main code
   3. documented as completely as the main code

In addition to installing matminer and automatminer, you will need to install
the benchdev package:
```bash
python setup_dev.py develop
```
You will also need to install the requirements in requirements-dev.txt

##### So don't expect a response or much help on the forum regarding dev (for now, at least.)
   
The available workflows include:
* nested CV benchmarks parallelized across folds (by node)
* multiple nested CV benchmarks (i.e., on many different data sets)
* plain ol' fitting operations (i.e., fitting a model for production)
   
Also, this `dev` folder is dependent upon specific environment variables, file
locations, and variables which are documented throughout the code. If you want
to run your own version of these dev tools, you'll need to set your own versions
of these variables specific to your computing platform, most of which you can
do through `config.py`.  

Finally, the results of these workflows are saved to a private database. If you
want to use your own databse, you'll need to substitute the code from the 
private database with your own (should only be a couple of lines from
`config.py` if you're familiar with pymongo and MongoDB). The general setup
of this database is as follows (by collection):

#### Automatminer results collections
- `automatminer_pipes`: individual MatPipes; most typically, folds
- `automatminer_benchmarks`: a collection of pipes, making up an entire benchmark (one complete ML result on one dataset by nested CV)
- `automatminer_builds`: a collection of benchmarks, making up an entire build (a collection of ML results on many datasets by nested CV)

#### Fireworks specific collections
- `fireworks`: the FireWorks collection containing individual fireworks (jobs)
- `workflows`: the FireWorks collection containing workflows of several jobs
- `launches`: the FireWorks collection containing info from different runs of jobs
- `fw_*`: FireWorks operations you probably don't need to worry about



