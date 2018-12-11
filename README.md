# automatminer
An automatic "black-box" yet interpretable prediction engine for materials properties.

##### Current code coverage:

[![CircleCI](https://circleci.com/gh/hackingmaterials/automatminer.svg?style=svg)](https://circleci.com/gh/hackingmaterials/automatminer)
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/aa63dd7aa85e480bbe0e924a02ad1540)](https://www.codacy.com/app/ardunn/automatminer?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=hackingmaterials/automatminer&amp;utm_campaign=Badge_Grade)
 [![Codacy Badge](https://api.codacy.com/project/badge/Coverage/aa63dd7aa85e480bbe0e924a02ad1540)](https://www.codacy.com/app/ardunn/automatminer?utm_source=github.com&utm_medium=referral&utm_content=hackingmaterials/automatminer&utm_campaign=Badge_Coverage)
 
 
### What is it?
Automatminer is a tool for automatically creating machine learning pipelines. Automatminer's pipelines include automatic featurization with matminer, feature reduction, and AutoML backend handling. Put in a materials dataset, get out a machine that predicts materials properties.
 
 
### What can it do?
Automatminer can make pipelines to accurately predict the properties from many kinds of materials data:
* both computational and experimental data
* small (~100 samples) to moderate (~100,000 samples) sized datasets
* crystalline datasets
* composition-only (i.e., unknown phases) datasets
* automatminer is agnostic to the target property, meaning it can be used to predict electronic, mechanical, thermodynamic, or any other kind of property


Automatminer automatically decorates a dataset using hundreds of descriptor techniques from matminer's descriptor library, picks the most useful features for learning, and runs a separate AutoML pipeline using TPOT. Once a pipeline has been fit, it can be examined with skater's interpretability tools, summarized in a text file, saved to disk, or used to make new predictions.  
 
### How do I use it?
The easiest (and most automatic) way to use automatminer is through the MatPipe object. First, fit the MatPipe to a dataframe containing materials objects such as chemical compositions (or pymatgen Structures) and some material target property.
```python
from automatminer.pipeline import MatPipe

# Fit a pipeline to training data to predict band gap
pipe = MatPipe()
pipe.fit(train_df, "band gap")
``` 

Now use your pipeline to predict the properties of some other data, such as a new composition or structure. 
```python
predicted_df = pipe.predict(other_df, "band gap")
```

You can also use it to benchmark against other machine learning models with the `benchmark` method of MatPipe, which optimizes the pipeline a training data and returns predictions on a held test set. 
```python
pipe = MatPipe()
test_predictions = pipe.benchmark(df, "bulk modulus", test_spec=0.2)
```

Once a MatPipe has been fit, you can examine it internally to see how it works using `pipe.digest()`; or pickle it for later with `pipe.save()`.


### How do I cite automatminer?
We are in the process of writing a paper for automatminer. In the meantime, please use the citation given in the matminer repo.


## Contributing 
Interested in contributing? See our [contribution guidelines](https://github.com/hackingmaterials/automatminer/blob/master/CONTRIBUTING.md) and make a pull request! Please submit questions, issues / bug reports, and all other communication through the [matminer Google Group](https://groups.google.com/forum/#!forum/matminer).
