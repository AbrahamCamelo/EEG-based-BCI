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
from sklearn.model_selection import train_test_split, KFold
from sklearn.pipeline import Pipeline
import numpy as np


logger = logging.getLogger(__name__)
logging.basicConfig(filename='logs/offline_acm_ts_svm_liu2024.log', level = logging.DEBUG)

acm_pipeline = Pipeline(
    steps = [('augmenteddataset',AugmentedDataset(order=8,lag=9)),
    ('covariances',Covariances(estimator='cov')),
    ('tangentspace',TangentSpace(metric='riemann')),
    ('svc',SVC(C=1.0, kernel='rbf'))]
)


time_now = datetime.now().strftime('%y-%m-%d %H:%M:%S')
logger.info(f'{time_now} Import dataset')


fmin, fmax = 8, 35
tmin, tmax = 0, None
events = ['left_hand', 'right_hand']


dataset = Liu2024()
paradigm = MotorImagery(events=events, n_classes=len(events), fmin=fmin, fmax=fmax, tmax=tmax)
X, y, metadata = paradigm.get_data(dataset)


subjects = metadata.subject.unique()

for subject in subjects:
    print(subject)

    s_index = metadata.subject == subject
    X_subject, y_subject = X[s_index], y[s_index]

    X_size = len(X_subject)
    training_portion = int(X_size/2)
    
    skf = KFold(n_splits=5,
                            shuffle=False,
                        )

    train_accs = []
    test_accs  = []
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X_subject, y_subject), start=1):
        x_train, x_test = X_subject[train_idx], X_subject[test_idx]
        y_train, y_test = y_subject[train_idx], y_subject[test_idx]

        acm_pipeline.fit(x_train, y_train)
        y_predict_train = acm_pipeline.predict(x_train)
        y_predict_test = acm_pipeline.predict(x_test)

        test_accuracy =  accuracy_score(y_test, y_predict_test)
        train_accuracy =  accuracy_score(y_train, y_predict_train)

        train_accs.append(train_accuracy)
        test_accs.append(test_accuracy)
        print(f"  Fold {fold_idx} — train_acc: {train_accuracy:.3f}, "
            f"test_acc: {test_accuracy:.3f}")
        
 

    testing_accuracy = np.mean(test_accs)


    
    
    training_accuracy = np.mean(train_accs)

    time_now = datetime.now().strftime('%y-%m-%d %H:%M:%S')
    logger.info(f'{time_now}    subject: {subject},  trainingAcc: {training_accuracy},   testingAcc: {testing_accuracy}')
