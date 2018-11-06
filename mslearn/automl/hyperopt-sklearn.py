from hpsklearn import HyperoptEstimator, any_classifier
from sklearn.datasets import fetch_mldata
from hyperopt import tpe
import numpy as np

"""

    pros:
        + super convenient and fast installation: pip install hpsklearn
    
    cons:
        - couldn't run any examples, for example this script copied from the examples
        in the documentation ( http://hyperopt.github.io/hyperopt-sklearn/ ) 
        returns the error:
        "ConnectionResetError: [Errno 54] Connection reset by peer"

"""

# Download the data and split into training and test sets

digits = fetch_mldata('MNIST original')

X = digits.data
y = digits.target

test_size = int( 0.2 * len( y ) )
np.random.seed(13)
indices = np.random.permutation(len(X))
X_train = X[ indices[:-test_size]]
y_train = y[ indices[:-test_size]]
X_test = X[ indices[-test_size:]]
y_test = y[ indices[-test_size:]]

estim = HyperoptEstimator( classifier=any_classifier('clf'),
                            algo=tpe.suggest, trial_timeout=300)

estim.fit( X_train, y_train )

print( estim.score( X_test, y_test ) )
# <<show score here>>
print( estim.best_model() )
# <<show model here>>