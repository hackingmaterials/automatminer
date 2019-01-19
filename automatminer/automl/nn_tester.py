import sys
sys.path.append("/Users/scherfaoui/LBL/automatminer")

from matminer.datasets.dataset_retrieval import load_dataset
from automatminer.automl.adaptors import NeuralNetworkAdaptor
from automatminer.pipeline import MatPipe
from automatminer.featurization import AutoFeaturizer
from automatminer.preprocessing import DataCleaner, FeatureReducer

autofeater = AutoFeaturizer()
cleaner = DataCleaner()
reducer = FeatureReducer()
learner = NeuralNetworkAdaptor()
pipe = MatPipe(learner=learner, autofeaturizer=autofeater, cleaner=cleaner, reducer=reducer)
df = load_dataset("elastic_tensor_2015")[["K_VRH", "structure"]]
predictions = pipe.benchmark(df[:-100], "K_VRH")
pipe.digest()
pipe.save("somefile.p")
pipe.load("somefile.p")
pipe.predict(df[-100:], "K_VRH")