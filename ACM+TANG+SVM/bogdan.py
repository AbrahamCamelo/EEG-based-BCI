import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from moabb.paradigms import MotorImagery

# moabb / pyriemann imports
from moabb.pipelines.features import AugmentedDataset
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from sklearn.model_selection import train_test_split
from moabb.datasets import Liu2024


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
    
def make_ACMTSSVM_pipeline():
    """
    Creates the pipeline as described by the YAML config.
    Returns a scikit-learn Pipeline object.
    """
    # steps = [
    #     # 1) Augmented Dataset
    #     ("augmenteddataset", AugmentedDataset(order=1, lag=1)),
    #     # 2) Covariance Estimation
    #     ("covariances", Covariances(estimator="cov")),
    #     # 3) Tangent Space
    #     ("tangentspace", TangentSpace(metric="riemann")),
    #     # 4) SVM Classifier
    #     ("svc", SVC(kernel="rbf")),
    # ]
    steps = [
        ("augmenteddataset", AugmentedDataset(order=4, lag=3)),
        ("covariances", Covariances(estimator="cov")),
        ("tangentspace", TangentSpace(metric="riemann")),
        ("svc", SVC(kernel="linear", C=0.1))
    ]
    pipeline = Pipeline(steps=steps)
    return pipeline


def get_param_grid():
    """
    Defines the parameter grid for hyperparameter optimization.
    Matches the 'param_grid' section of the YAML.
    """
    # param_grid = {
    #     "augmenteddataset__order": list(range(1, 11)),  # 1 to 10
    #     "augmenteddataset__lag": list(range(1, 11)),    # 1 to 10
    #     "svc__C": [0.5, 1, 1.5],
    #     "svc__kernel": ["rbf", "linear"],
    # }
    param_grid = {
        "augmenteddataset__order": list(range(1, 5)),  # 1 to 10
        "augmenteddataset__lag": list(range(1, 5)),    # 1 to 10
        "svc__C": [0.1, 0.5, 1, 1.5],
        "svc__kernel": ["rbf", "linear"],
    }
    return param_grid

def main_without_grid_search(subjects):
    """
    Example main function to show how to instantiate the pipeline and fit it
    using default parameters, without performing a grid search.
    """
    res = {}
    for subject in subjects:
        print(f"Subject {subject}:")
        
        # Retrieve the data for the subject
        X, y, info = get_data(subjects=[subject], get_info=True, tmin=0, tmax=None, resample_to=125)
        # Downsample: take every second element along the middle dimension
        # X = X[:, np.arange(0, X.shape[1], 2), :]

        # Create the pipeline with default parameters
        pipeline = make_ACMTSSVM_pipeline()

        # Split the data into training and testing sets
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.5, stratify=None, shuffle=False)

        # Fit the pipeline on the training data
        pipeline.fit(x_train, y_train)

        # Predict on both the training and testing sets
        y_pred_train = pipeline.predict(x_train)
        y_pred = pipeline.predict(x_test)

        # Calculate accuracy scores
        accuracy_train = accuracy_score(y_train, y_pred_train)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Train accuracy: {accuracy_train:.4f}, Test accuracy: {accuracy:.4f}")
        
        res[subject] = (accuracy_train, accuracy)
    
    # Summarize results for all subjects
    print("Subject, train_acc, test_acc")
    for subject, (train_acc, test_acc) in res.items():
        print(f"{subject}: {round(train_acc, 2)}, {round(test_acc, 2)}")
    
    # Compute and print the average test accuracy over subjects
    avg_test_acc = np.mean([test_acc for (_, test_acc) in res.values()])
    print(f"Average test_acc: {round(avg_test_acc, 2)}")



def main_with_grid_search(subjects):
    """
    Example main function to show how to instantiate the pipeline and do a GridSearchCV.
    """

    res = {}
    best_params = {}
    for subject in subjects:
        print(f"Subject {subject}:")
        X, y, info = get_data(subjects=[subject], get_info=True, tmin=0, tmax=None, resample_to=125)
        X = X[:,np.arange(0, X.shape[1], 2), :]
        
        pipeline = make_ACMTSSVM_pipeline()
        param_grid = get_param_grid()

        # Example: setting up a GridSearchCV
        # (Typically, you'd also define a cross-validation scheme.)
        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            scoring="accuracy",
            cv=10,  # Or another CV strategy
            n_jobs=-1,
            verbose=1
        )
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.5,  stratify=None, shuffle=False)

        

        print("Starting GridSearchCV fitting...")
        grid_search.fit(x_train, y_train)
        print("Best params found:", grid_search.best_params_)
        print("Best cross-validation score:", grid_search.best_score_)
        # mean_val_accuracy = grid_search.cv_results_['mean_test_score'][grid_search.best_index_]
        # print(f"Mean validation accuracy over folds: {mean_val_accuracy:.4f}")

        pipeline.set_params(**grid_search.best_params_)
        best_params[subject] = grid_search.best_params_
        pipeline.fit(x_train, y_train)
        y_pred_train = pipeline.predict(x_train)
        y_pred = pipeline.predict(x_test)
        accuracy_train = accuracy_score(y_train, y_pred_train)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Train accuracy: {accuracy_train:.4f}, Test accuracy: {accuracy:.4f}")
        res[subject] = (accuracy_train, accuracy)
    print("Subject, train_acc, test_acc")
    for entry in res:
        print(f"{entry}: {round(res[entry][0],2)}, {round(res[entry][1],2)}, {best_params[entry]}")
        
    print(f"Average test_acc: {round(np.mean([res[entry][1] for entry in res]),2)}")
    



if __name__ == "__main__":
    main_without_grid_search(subjects=range(1, 51))
    