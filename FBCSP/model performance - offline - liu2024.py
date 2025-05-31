from moabb.datasets import Liu2024
from mne import events_from_annotations
from moabb.paradigms import MotorImagery
from mne import Epochs
import numpy as np
from fbcsp import MLEngine

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold


dataset = Liu2024()

''' Function ot obtain the raw epochs from a mne.io.Raw object '''
def get_data(subjects, fmin=4, fmax=40, tmin=0, tmax=None, get_info=False, resample_to=None):
    events = ['left_hand', 'right_hand']
    
    dataset = Liu2024()
    paradigm = MotorImagery(events=events, n_classes=len(events), fmin=fmin, fmax=fmax, tmin=tmin, tmax=tmax, resample=resample_to)
    X, y, metadata = paradigm.get_data(dataset, subjects=subjects)
    y = np.array(list(map(lambda x: events.index(x), y)))
    if get_info:
        sessions = dataset.get_data(subjects=subjects)
        session_name = list(sessions.keys())[0]  # e.g., 'session_0'
        run_name = list(sessions[session_name].keys())[0]  # e.g., 'run_0'
        raw = sessions[session_name][run_name]
        channel_names = raw['0'].ch_names
        sfreq = raw['0'].info['sfreq']
        info = {'channel_names': channel_names, 'sfreq': sfreq}
        return X, y, info
    else:
        return X, y

res = dict()


for subject in range(1,51):
    print(f"Subject {subject}:")
    X, y, info = get_data(subjects=[subject], get_info=True, tmin=0, tmax=None, resample_to=125)
    X = X[:,np.arange(0, X.shape[1], 2), :]
    mle = MLEngine(m_filters=2, feature_selection=True, fs=125)
    skf = StratifiedKFold(5, shuffle=False)
    train_acc =[]
    test_acc = []
    for i, (train_index, test_index) in enumerate(skf.split(X, y)):
        X_train, X_val = X[train_index], X[test_index]
        y_train, y_val = y[train_index], y[test_index]
        results = mle.experiment(X_train, y_train, X_val, y_val)
        train_acc.append(results['train_acc'])
        test_acc.append(results['test_acc'])
    
    res[subject] = (np.mean(train_acc),np.mean(test_acc))



print('accuracy of each subject')
print('subject, training accuracy, testing accuracy')
for key, value in dict(sorted(res.items())).items():
    print(f"{str(key)}, {round(value[0],3)}, {round(value[1],3)}")