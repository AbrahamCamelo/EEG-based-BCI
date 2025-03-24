import numpy as np
from moabb.paradigms import MotorImagery
from moabb.datasets import Liu2024
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from sklearn.svm import SVC
from moabb.pipelines.features import AugmentedDataset
from sklearn.metrics import accuracy_score
import logging
from datetime import datetime

logger = logging.getLogger(__name__)
logging.basicConfig(filename='logs/online_acm_ts_svm_liu2024.log', level = logging.DEBUG)

fmin, fmax = 8, 35
tmin, tmax = 0, None
events = ['left_hand', 'right_hand']

time_now = datetime.now().strftime('%y-%m-%d %H:%M:%S')
logger.info(f'{time_now} Import dataset')


dataset = Liu2024()

paradigm = MotorImagery(events=events, n_classes=len(events), fmin=fmin, fmax=fmax, tmax=tmax)

X, y, metadata = paradigm.get_data(dataset)

order, lag = 8, 9
C_value, kernel_type = 1.0, "rbf"

time_now = datetime.now().strftime('%y-%m-%d %H:%M:%S')
logger.info(f'{time_now}    Augmented Dataset')

augmented_dataset = AugmentedDataset(order=order, lag=lag)
X_augmented = augmented_dataset.fit_transform(X, y)

time_now = datetime.now().strftime('%y-%m-%d %H:%M:%S')
logger.info(f'{time_now}    Covariance Calculation')

cov_estimator = Covariances("cov")
X_cov = cov_estimator.fit_transform(X_augmented)


for subject in dataset.subject_list:
    
    subject_index = metadata['subject'] == subject

    X_cov_subject = X_cov[subject_index]

    training_portion = int(len(X_cov_subject)/2)
    print(training_portion)
    X_cov_train = X_cov_subject[:training_portion]
    y_train = y[subject_index][:training_portion]

    X_cov_test = X_cov_subject[training_portion:]
    y_test = y[subject_index][training_portion:]


    svm_classifier = SVC(C=C_value, kernel=kernel_type)

    predict_list = list()

    time_now = datetime.now().strftime('%y-%m-%d %H:%M:%S')
    logger.info(f'{time_now}    Beginning of SVM Test, subject {subject}')

    for i_test in range(len(y_test)):
        
        tangent_space = TangentSpace(metric="riemann")
        X_train = np.concatenate((X_cov_train, X_cov_test[:i_test]))
        X_test = np.array([X_cov_test[i_test]])
        
        tangent_space.fit(X_train)
        X_tgsp_train = tangent_space.transform(X_train)
        X_tgsp_test = tangent_space.transform(X_test)

        y_train_concat = np.concatenate((y_train,y_test[:i_test]))
        svm_classifier.fit(X_tgsp_train, y_train_concat)
        
        y_predict_train = svm_classifier.predict(X_tgsp_train)
        y_predict = svm_classifier.predict(X_tgsp_test.reshape(1,-1))

        predict_list.append(y_predict[0])
        actual_acc = accuracy_score(y_test[:len(predict_list)],
                                    predict_list)
        
        training_accuracy = accuracy_score(y_predict_train,
                                    y_train_concat)

        time_now = datetime.now().strftime('%y-%m-%d %H:%M:%S')
        logger.info(f'{time_now}    subject: {subject},   trial: {i_test + 1},    real class: {y_test[i_test]},   predicted class: {y_predict[0]},  val: {y_test[i_test] == y_predict[0]},  trainingAcc: {training_accuracy},   actualAcc: {actual_acc}')
