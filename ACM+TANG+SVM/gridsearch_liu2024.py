from moabb.paradigms import MotorImagery
from moabb.datasets import Liu2024
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from sklearn.svm import SVC
from moabb.pipelines.features import AugmentedDataset
import logging
import warnings


from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

import pandas as pd

logging.basicConfig(filename='logs/gridsearch_acm_ts_svm_liu2024.log', level = logging.DEBUG)
logging.captureWarnings(True)

acm_pipeline = Pipeline(
    steps = [('augmenteddataset',AugmentedDataset(lag=5, order=4)),
    ('covariances',Covariances(estimator='cov')),
    ('tangentspace',TangentSpace(metric='riemann')),
    ('svc',SVC(C=1.0, kernel='rbf'))]
)


fmin, fmax = 8, 35
tmin, tmax = 0, None
events = ['left_hand', 'right_hand']

dataset = Liu2024()
paradigm = MotorImagery(events=events, n_classes=len(events), fmin=fmin, fmax=fmax, tmax=tmax)
X, y, metadata = paradigm.get_data(dataset)


param_grid = [{
    'augmenteddataset__lag':list(range(1,6)),
    'augmenteddataset__order':list(range(1,6)),
    'svc__C':list(range(1,10))
    }
]

gridsearch = GridSearchCV(acm_pipeline ,param_grid[0], cv = 5, n_jobs=-1, verbose=2, scoring='accuracy')

df_params = pd.DataFrame(columns=['augmenteddataset__lag','augmenteddataset__order','svc__C','subject','average_accuracy'])

subjects = metadata.subject.unique()
for subject in subjects:
    print(subject)

    s_index = metadata.subject == subject
    X_subject, y_subject = X[s_index], y[s_index]

    gridsearch.fit(X_subject, y_subject)

    best_params = pd.DataFrame([gridsearch.best_params_])
    best_params['subject'] = subject
    best_params['average_accuracy'] = gridsearch.best_score_

    df_params = pd.concat([df_params,best_params],ignore_index=True)

df_params.to_csv('best_params_liu2024.csv', index=False)