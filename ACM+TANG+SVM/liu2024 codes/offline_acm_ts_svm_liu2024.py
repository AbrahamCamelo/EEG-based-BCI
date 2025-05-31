from moabb.paradigms import MotorImagery
from moabb.datasets import Liu2024
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from sklearn.svm import SVC
from moabb.pipelines.features import AugmentedDataset
import logging
from datetime import datetime
from sklearn.metrics import accuracy_score
import logging
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import numpy as np


logger = logging.getLogger(__name__)
logging.basicConfig(filename='logs/offline_acm_ts_svm_liu2024.log', level = logging.DEBUG)

acm_pipeline = Pipeline(
    steps = [('augmenteddataset',AugmentedDataset(order=4,lag=3)),
    ('covariances',Covariances(estimator='cov')),
    ('tangentspace',TangentSpace(metric='riemann')),
    ('svc',SVC(C=0.5, kernel='linear'))]
    
)


time_now = datetime.now().strftime('%y-%m-%d %H:%M:%S')
logger.info(f'{time_now} Import dataset')


fmin, fmax = 8, 35
tmin, tmax = 0, None
events = ['left_hand', 'right_hand']


dataset = Liu2024()
paradigm = MotorImagery(events=events, n_classes=len(events), fmin=fmin, fmax=fmax, tmax=tmax)
X, y, metadata = paradigm.get_data(dataset)
X = X[:, np.arange(0, X.shape[1], 2), :]

subjects = metadata.subject.unique()

for subject in subjects:
    print(subject)

    s_index = metadata.subject == subject
    X_subject, y_subject = X[s_index], y[s_index]
   
    
    X_subject_train, X_subject_test, y_subject_train, y_subject_test = train_test_split(X_subject,y_subject, shuffle=False, test_size=0.5)

 
    acm_pipeline.fit(X_subject_train, y_subject_train)
    
    y_predict_train = acm_pipeline.predict(X_subject_train)
    y_predict_test = acm_pipeline.predict(X_subject_test)

    testing_accuracy = accuracy_score(y_subject_test, y_predict_test)
    
    training_accuracy = accuracy_score(y_subject_train, y_predict_train)

    logger.info(f'trainingAcc: {training_accuracy},   testingAcc: {testing_accuracy}')
