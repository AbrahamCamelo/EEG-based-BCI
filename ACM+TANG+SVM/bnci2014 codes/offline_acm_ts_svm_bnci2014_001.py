import numpy as np
from moabb.paradigms import MotorImagery
from moabb.datasets import BNCI2014_001
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from sklearn.svm import SVC
from moabb.pipelines.features import AugmentedDataset
import logging
from datetime import datetime
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


logger = logging.getLogger(__name__)
logging.basicConfig(filename='logs/offline_acm_ts_svm_bnci2014_001.log', level = logging.DEBUG)

acm_pipeline = Pipeline(
    steps = [('augmenteddataset',AugmentedDataset(order=8,lag=9)),
    ('covariances',Covariances(estimator='cov')),
    ('tangentspace',TangentSpace(metric='riemann')),
    ('svc',SVC(C=1.0, kernel='rbf'))]
)


fmin, fmax = 8, 35
tmin, tmax = 0, None
events = ['feet', 'left_hand', 'right_hand', 'tongue']

time_now = datetime.now().strftime('%y-%m-%d %H:%M:%S')
logger.info(f'{time_now} Import dataset')

dataset = BNCI2014_001()
paradigm = MotorImagery(events=events, n_classes=len(events), fmin=fmin, fmax=fmax, tmax=tmax)
X, y, metadata = paradigm.get_data(dataset)


for subject in dataset.subject_list:
    print(subject)
    
    subject_index = metadata['subject'] == subject
    train_index = metadata[subject_index]['session'] == '0train'
    test_index = metadata[subject_index]['session'] != '0train'

    X_subject_train, X_subject_test = X[subject_index][train_index], X[subject_index][test_index]
    y_subject_train, y_subject_test = y[subject_index][train_index], y[subject_index][test_index]
 

    acm_pipeline.fit(X_subject_train, y_subject_train)
    
    y_predict_train = acm_pipeline.predict(X_subject_train)
    y_predict_test = acm_pipeline.predict(X_subject_test)

    testing_accuracy = accuracy_score(y_subject_test, y_predict_test)
    
    training_accuracy = accuracy_score(y_subject_train, y_predict_train)

    time_now = datetime.now().strftime('%y-%m-%d %H:%M:%S')
    logger.info(f'{time_now}    subject: {subject},  trainingAcc: {training_accuracy},   testingAcc: {testing_accuracy}')
