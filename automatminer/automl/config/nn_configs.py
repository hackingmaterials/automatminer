# """
# Default arguments and configuration for the Neural Network adaptor (NNAdaptor).
# """
#
#
# NN_DEFAULT_CONFIG = {
#     "init": "glorot_uniform",
#     "optimizer": "adam",
#     "hidden_layer_sizes": (3, 50, 100),
#     "units": 20,
#     "dropout": 0.5,
#     "batch_spec": ((400, 1024), (100, -1)),
#     "activation": "sigmoid",
#     "input_noise": 0,
#     "use_maxout": False,
# }
#
# init="glorot_uniform", optimizer="adam",
#                  hidden_layer_sizes=2, units=20, dropout=0.5,
#                  show_accuracy=True, batch_spec=((400, 1024), (100, -1)),
#                  activation="sigmoid", input_noise=0., use_maxout=False,
#                  use_maxnorm=False, learning_rate=0.001, stop_early=False,
#                  kfold_splits=2, mode="regression"
#
