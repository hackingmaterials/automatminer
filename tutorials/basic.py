"""
An example of basic usage of Automatminer.

For more examples, please see the other files in the /examples folder.
"""

from automatminer import get_preset_config
from automatminer import MatPipe

# The most basic usage of automatminer requires interacting with only one class,
# MatPipe. This class, once fit, is a complete pipeline, and is able to
# transform compositions, structures, bandstructures, and DOS into property
# predictions.

# A configured MatPipe object will featurize, clean, and learn on a dataset
# automatically, and it made of 4 classes: AutoFeaturizer, DataCleaner,
# FeatureReducer, and an ML adaptor (e.g., TPOTAdaptor). The exact operations
# MatPipe executes are based entirely on how these 4 classes are configured.

# The easiest way to get started is by passing in a preset configuration to
# MatPipe. We can do this with the get_preset_config function; here, we'll use
# the "express" config, which will provide decent results in a reasonable time
# frame (an hour or two).
pipe = MatPipe(**get_preset_config("debug_single"))

# Let's download an example dataset and try predicting bulk moduli.
from sklearn.model_selection import train_test_split
from matminer.datasets.dataset_retrieval import load_dataset
df = load_dataset("elastic_tensor_2015")[["structure", "K_VRH"]]
train, test = train_test_split(df, shuffle=True, random_state=20190301, test_size=0.2)
test_true = test['K_VRH']
test = test.drop(columns=["K_VRH"])

# MatPipe uses an sklearn-esque BaseEstimator API for fitting pipelines and
# predicting properties. Fitting a pipe trains it to the input data; predicting
# with a pipe will output predictions.
pipe.fit(train, target="K_VRH")

# Now we can predict our outputs. They'll appear in a column called
# "K_VRH predicted".
test_predicted = pipe.predict(test, "K_VRH")["K_VRH predicted"]

# Let's see how we did:
from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(test_true, test_predicted)
print("MAE on {} samples: {}".format(len(test_true), mae))

# Save a text digest of the pipeline.
pipe.digest(filename="digest.txt")

# You can now save your model
pipe.save("mat.pipe")

# Then load it later and make predictions on new data
pipe_loaded = MatPipe.load("mat.pipe")

# You have reached the end of the basic tutorial. Please see the other tutorials
# or the online documentation for more info!



