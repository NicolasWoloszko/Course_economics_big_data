import warnings
import numbers
import time
from traceback import format_exception_only

import numpy as np
import scipy.sparse as sp

from sklearn.base import is_classifier, clone
from sklearn.utils import indexable
from sklearn.utils.validation import _is_arraylike, _num_samples
from sklearn.utils.metaestimators import _safe_split
from sklearn.utils._joblib import Parallel, delayed
from sklearn.metrics.scorer import check_scoring, _check_multimetric_scoring
from sklearn.exceptions import FitFailedWarning
from sklearn.model_selection._split import check_cv
from sklearn.preprocessing import LabelEncoder

import pandas as pd

from sklearn.model_selection._validation import _fit_and_predict

def cross_val_predict(estimator, X, y=None, groups=None, cv='warn',
                      n_jobs=None, verbose=0, fit_params=None,
                      pre_dispatch='2*n_jobs', method='predict'):

    """
    Minor modifications and simplications brought to the sklearn function in order to allow
    for application with non-partition CV scheme. 
    """

    X, y, groups = indexable(X, y, groups)

    cv = check_cv(cv, y, classifier=is_classifier(estimator))

    # We clone the estimator to make sure that all the folds are
    # independent, and that it is pickle-able.
    parallel = Parallel(n_jobs=n_jobs, verbose=verbose,
                        pre_dispatch=pre_dispatch)
    prediction_blocks = parallel(delayed(_fit_and_predict)(
        clone(estimator), X, y, train, test, verbose, fit_params, method)
        for train, test in cv.split(X, y, groups))


    # Concatenate the predictions
    predictions = [pred_block_i for pred_block_i, _ in prediction_blocks]
    predictions = np.concatenate(predictions)

    test_indices = np.concatenate([indices_i
                                   for _, indices_i in prediction_blocks])
    test_index = [y.index[_] for _ in test_indices]
    #print(predictions)

    if y.ndim == 1:
        return pd.Series(predictions, index = test_index)
    elif y.ndim>1:
        return pd.DataFrame(predictions, index = test_index)
    
