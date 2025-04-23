from moabb.datasets import BNCI2014_001
from mne import events_from_annotations
from mne import Epochs
import numpy as np
from fbcsp import MLEngine

from sklearn.model_selection import train_test_split


dataset = BNCI2014_001()
data = dataset.get_data()


''' Function ot obtain the raw epochs from a mne.io.Raw object '''
def get_data_from_run(raw_run):
    events_id = {'feet': 0, 'left_hand': 1, 'right_hand': 2, 'tongue': 3}
    events, event_id = events_from_annotations(raw_run, event_id=events_id)
    epochs = Epochs(raw_run, events, event_id=event_id, tmin=0.0, tmax=4.0, preload=True, baseline=None, verbose=0)
    X = epochs.get_data(picks=['eeg'])
    y = epochs.events[:,2]

    return X, y


''' Training and testing '''
X = np.zeros((48*12,22,1001), float)
y = np.zeros((48*12,), int)

res = dict()

for subject in dataset.subject_list:
    raw_subject = data[subject]
    
    ''' Since each session of a subject has 6 runs, here we join the 6 runs in a single X (total epochs = 48*6)'''
    for i, session in enumerate(raw_subject):
        for j, run in enumerate(raw_subject[session].values()):
            X_run, y_run = get_data_from_run(run)

            session_index = i*6*48
            run_index = 48*j
            X[run_index + session_index : run_index + 48 + session_index, :, :] = X_run
            y[run_index + session_index : run_index + 48 + session_index] = y_run


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
    model = MLEngine(m_filters=2, fs = 250)
    results = model.experiment(X_train, y_train, X_test, y_test)
    res[subject] = (results['train_acc'], results['test_acc'])


print('accuracy of each subject')
print('subject, training accuracy, testing accuracy')
for key, value in dict(sorted(res.items())).items():
    print(f"{str(key)}, {round(value[0],3)}, {round(value[1],3)}")
 
