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



logger = logging.getLogger(__name__)
logging.basicConfig(filename='logs/online_acm_ts_svm_liu2024.log', level = logging.DEBUG)

acm_pipeline = Pipeline(
    steps = [('augmenteddataset',AugmentedDataset(order=8,lag=9)),
    ('covariances',Covariances(estimator='cov')),
    ('tangentspace',TangentSpace(metric='riemann')),
    ('svc',SVC(C=1.0, kernel='rbf'))]
)

fmin, fmax = 8, 35
tmin, tmax = 0, None
events = ['left_hand', 'right_hand']

time_now = datetime.now().strftime('%y-%m-%d %H:%M:%S')
logger.info(f'{time_now} Import dataset')

dataset = Liu2024()
paradigm = MotorImagery(events=events, n_classes=len(events), fmin=fmin, fmax=fmax, tmax=tmax)
X, y, metadata = paradigm.get_data(dataset)



subjects = metadata.subject.unique()

for subject in subjects[15:]:
    print(subject)

    s_index = metadata.subject == subject
    X_subject, y_subject = X[s_index], y[s_index]

    X_size = len(X_subject)
    
    predict_list = list()

    time_now = datetime.now().strftime('%y-%m-%d %H:%M:%S')
    logger.info(f'{time_now}    Beginning of SVM Test, subject {subject}')

    for i_test in range(int(X_size/2), X_size):
        
        X_subject_train, y_subject_train = X_subject[:i_test], y_subject[:i_test]
        X_subject_test, y_subject_test = X_subject[i_test], y_subject[i_test]
        
        acm_pipeline.fit(X_subject_train, y_subject_train)

        y_predict_train = acm_pipeline.predict(X_subject_train)
        y_predict = acm_pipeline.predict(np.array([X_subject_test]))

        predict_list.append(y_predict[0])

        actual_acc = accuracy_score(y_subject[int(X_size/2):i_test+1],
                                    predict_list)
        
        training_accuracy = accuracy_score(y_predict_train,
                                    y_subject_train)
        

        time_now = datetime.now().strftime('%y-%m-%d %H:%M:%S')
        logger.info(f'{time_now}    subject: {subject},   trial: {i_test - int(X_size/2)},    real class: {y_subject_test},   predicted class: {y_predict[0]},  val: {y_subject_test == y_predict[0]},  trainingAcc: {training_accuracy},   actualAcc: {actual_acc}')
