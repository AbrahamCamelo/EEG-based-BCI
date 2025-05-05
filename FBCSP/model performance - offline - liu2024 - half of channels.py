from moabb.datasets import Liu2024
from mne import events_from_annotations
from mne import Epochs
import numpy as np
from fbcsp import MLEngine
from moabb.paradigms import MotorImagery

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold

dataset = Liu2024()
data = dataset.get_data()


''' Function ot obtain the raw epochs from a mne.io.Raw object '''
def get_data_from_run(raw_run):
    events_id = {'left_hand': 0, 'right_hand': 1}
    events, event_id = events_from_annotations(raw_run, event_id=events_id)
    epochs = Epochs(raw_run, events, event_id=event_id, tmin=0.0, tmax=4.0, preload=True, baseline=None, verbose=0)
    epochs.resample(125)
    X = epochs.get_data(picks=['eeg'])
    X = X[:,np.arange(0, X.shape[1], 2), :]
    y = epochs.events[:,2]

    return X, y

res = dict()

for subject in data:
    for session in data[subject]:
        for run in data[subject][session].values():
            X, y = get_data_from_run(run)

    skf = StratifiedKFold(5, shuffle=True, random_state=42)
    train_acc =[]
    test_acc = []
    for i, (train_index, test_index) in enumerate(skf.split(X, y)):
        X_train, X_val = X[train_index], X[test_index]
        y_train, y_val = y[train_index], y[test_index]
        model = MLEngine(m_filters=2, fs = 125)
        results = model.experiment(X_train, y_train, X_val, y_val)
        train_acc.append(results['train_acc'])
        test_acc.append(results['test_acc'])
    
    res[subject] = (np.mean(train_acc),np.mean(test_acc))


print('accuracy of each subject')
print('subject, training accuracy, testing accuracy')
for key, value in dict(sorted(res.items())).items():
    print(f"{str(key)}, {round(value[0],3)}, {round(value[1],3)}")
