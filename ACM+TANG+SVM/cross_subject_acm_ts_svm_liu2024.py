from moabb.paradigms import MotorImagery
from moabb.datasets import Liu2024
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from sklearn.svm import SVC
from moabb.pipelines.features import AugmentedDataset
import logging
from datetime import datetime
import numpy as np
from sklearn.metrics import accuracy_score
import logging

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split




acm_pipeline = Pipeline(
    steps = [('augmenteddataset',AugmentedDataset(order=3,lag=2)),
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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


acm_pipeline.fit(X_train, y_train)

y_predict_train = acm_pipeline.predict(X_train)
y_predict = acm_pipeline.predict(X_test)

test_acc = accuracy_score(y_predict,
                            y_test)

training_accuracy = accuracy_score(y_predict_train,
                            y_train)
print(training_accuracy,test_acc)


