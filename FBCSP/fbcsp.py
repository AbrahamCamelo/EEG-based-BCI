import numpy as np
from scipy import signal
from scipy.signal import cheb2ord
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split

import scipy

class FilterBank:
    def __init__(self, fs):
        self.fs = fs
        self.f_trans = 2
        self.f_pass = np.arange(4, 40, 4)
        self.f_width = 4
        self.gpass = 3
        self.gstop = 30
        self.filter_coeff = {}

    def get_filter_coeff(self):
        Nyquist_freq = self.fs / 2

        for i, f_low_pass in enumerate(self.f_pass):
            f_pass = np.asarray([f_low_pass, f_low_pass + self.f_width])
            f_stop = np.asarray([f_pass[0] - self.f_trans, f_pass[1] + self.f_trans])
            wp = f_pass / Nyquist_freq
            ws = f_stop / Nyquist_freq
            order, wn = cheb2ord(wp, ws, self.gpass, self.gstop)
            b, a = signal.cheby2(order, self.gstop, ws, btype="bandpass")
            self.filter_coeff.update({i: {"b": b, "a": a}})
        return self.filter_coeff

    def filter_data(self, eeg_data, window_details=None):
        n_trials, n_channels, n_samples = eeg_data.shape
        if window_details:
            n_samples = int(self.fs*(window_details.get('tmax')-window_details.get('tmin')))+1
        
        filtered_data = np.zeros(
            (len(self.filter_coeff), n_trials, n_channels, n_samples)
        )
        for i, fb in self.filter_coeff.items():
            b = fb.get("b")
            a = fb.get("a")
            eeg_data_filtered = np.asarray(
                [signal.lfilter(b, a, eeg_data[j, :, :]) for j in range(n_trials)]
            )
            if window_details is not None:
                eeg_data_filtered = eeg_data_filtered[:,:,int((window_details.get('tmin'))*self.fs):int((window_details.get('tmax'))*self.fs)+1]
            filtered_data[i, :, :, :] = eeg_data_filtered
        return filtered_data


class CSP:
    def __init__(self, m_filters):
        self.m_filters = m_filters

    def fit(self, x_train, y_train):
        x_data = np.copy(x_train)
        y_labels = np.copy(y_train)
        n_trials, n_channels, n_samples = x_data.shape
        cov_x = np.zeros((2, n_channels, n_channels), dtype=float)
        for i in range(n_trials):
            x_trial = x_data[i, :, :]
            y_trial = y_labels[i]
            cov_x_trial = np.matmul(x_trial, np.transpose(x_trial))
            cov_x_trial /= np.trace(cov_x_trial)
            cov_x[y_trial, :, :] += cov_x_trial

        cov_x = np.asarray([cov_x[cls] / np.sum(y_labels == cls) for cls in range(2)])
        cov_combined = cov_x[0] + cov_x[1]
        eig_values, u_mat = scipy.linalg.eig(cov_combined, cov_x[0])
        sort_indices = np.argsort(abs(eig_values))[::-1]
        eig_values = eig_values[sort_indices]
        u_mat = u_mat[:, sort_indices]
        u_mat = np.transpose(u_mat)

        return eig_values, u_mat

    def transform(self, x_trial, eig_vectors):
        z_trial = np.matmul(eig_vectors, x_trial)
        z_trial_selected = z_trial[: self.m_filters, :]
        z_trial_selected = np.append(
            z_trial_selected, z_trial[-self.m_filters :, :], axis=0
        )
        sum_z2 = np.sum(z_trial_selected**2, axis=1)
        sum_z = np.sum(z_trial_selected, axis=1)
        var_z = (sum_z2 - (sum_z**2) / z_trial_selected.shape[1]) / (
            z_trial_selected.shape[1] - 1
        )
        sum_var_z = sum(var_z)
        return np.log(var_z / sum_var_z)


class FBCSP:
    def __init__(self, m_filters):
        self.m_filters = m_filters
        self.fbcsp_filters_multi = []

    def fit(self, x_train_fb, y_train):
        y_classes_unique = np.unique(y_train)
        n_classes = len(y_classes_unique)
        self.csp = CSP(self.m_filters)

        def get_csp(x_train_fb, y_train_cls):
            fbcsp_filters = {}
            for j in range(x_train_fb.shape[0]):
                x_train = x_train_fb[j, :, :, :]
                eig_values, u_mat = self.csp.fit(x_train, y_train_cls)
                fbcsp_filters.update({j: {"eig_val": eig_values, "u_mat": u_mat}})
            return fbcsp_filters

        for i in range(n_classes):
            cls_of_interest = y_classes_unique[i]
            select_class_labels = lambda cls, y_labels: [
                0 if y == cls else 1 for y in y_labels
            ]
            y_train_cls = np.asarray(select_class_labels(cls_of_interest, y_train))
            fbcsp_filters = get_csp(x_train_fb, y_train_cls)
            self.fbcsp_filters_multi.append(fbcsp_filters)

    def transform(self, x_data, class_idx=0):
        n_fbanks, n_trials, n_channels, n_samples = x_data.shape
        x_features = np.zeros(
            (n_trials, self.m_filters * 2 * len(x_data)), dtype=float
        )
        for i in range(n_fbanks):
            
            eig_vectors = self.fbcsp_filters_multi[class_idx].get(i).get("u_mat")
            # eig_values = self.fbcsp_filters_multi[class_idx].get(i).get('eig_val')
            for k in range(n_trials):
                x_trial = np.copy(x_data[i, k, :, :])
                csp_feat = self.csp.transform(x_trial, eig_vectors)
                for j in range(self.m_filters):
                    x_features[k, i * self.m_filters * 2 + (j + 1) * 2 - 2] = csp_feat[j]
                    x_features[k, i * self.m_filters * 2 + (j + 1) * 2 - 1] = csp_feat[-j - 1]

        return x_features
    
    
class Classifier:
    def __init__(self,model, features_selection=True):
        self.model = model
        self.feature_selection = features_selection

    def predict(self,x_features):
        if self.feature_selection:
            x_features_selected = self.feature_selector.transform(x_features)
        else:
            x_features_selected = x_features
        y_predicted = self.model.predict(x_features_selected)
        return y_predicted
    

    def fit(self,x_features,y_train):
        if self.feature_selection:
            self.feature_selector = FeatureSelect()
            x_train_features_selected = self.feature_selector.fit(x_features,y_train)
        else:
            x_train_features_selected = x_features
        self.model.fit(x_train_features_selected,y_train)
        y_predicted = self.model.predict(x_train_features_selected)
        return y_predicted


class FeatureSelect:
    def __init__(self, n_features_select=4, n_csp_pairs=2):
        self.n_features_select = n_features_select
        self.n_csp_pairs = n_csp_pairs
        self.features_selected_indices=[]

    def fit(self,x_train_features,y_train):
        MI_features = self.MIBIF(x_train_features, y_train)
        MI_sorted_idx = np.argsort(MI_features)[::-1]
        features_selected = MI_sorted_idx[:self.n_features_select]

        paired_features_idx = self.select_CSP_pairs(features_selected, self.n_csp_pairs)
        x_train_features_selected = x_train_features[:, paired_features_idx]
        self.features_selected_indices = paired_features_idx

        return x_train_features_selected

    def transform(self,x_test_features):
        return x_test_features[:,self.features_selected_indices]

    def MIBIF(self, x_features, y_labels):
        def get_prob_pw(x,d,i,h):
            n_data = d.shape[0]
            t=d[:,i]
            kernel = lambda u: np.exp(-0.5*(u**2))/np.sqrt(2*np.pi)
            prob_x = 1 / (n_data * h) * sum(kernel((np.ones((len(t)))*x- t)/h))
            return prob_x

        def get_pd_pw(d, i, x_trials):
            n_data, n_dimensions = d.shape
            if n_dimensions==1:
                i=1
            t = d[:,i]
            min_x = np.min(t)
            max_x = np.max(t)
            n_trials = x_trials.shape[0]
            std_t = np.std(t)
            if std_t==0:
                h=0.005
            else:
                h=(4./(3*n_data))**(0.2)*std_t
            prob_x = np.zeros((n_trials))
            for j in range(n_trials):
                prob_x[j] = get_prob_pw(x_trials[j],d,i,h)
            return prob_x, x_trials, h

        y_classes = np.unique(y_labels)
        n_classes = len(y_classes)
        n_trials = len(y_labels)
        prob_w = []
        x_cls = {}
        for i in range(n_classes):
            cls = y_classes[i]
            cls_indx = np.where(y_labels == cls)[0]
            prob_w.append(len(cls_indx) / n_trials)
            x_cls.update({i: x_features[cls_indx, :]})

        prob_x_w = np.zeros((n_classes, n_trials, x_features.shape[1]))
        prob_w_x = np.zeros((n_classes, n_trials, x_features.shape[1]))
        h_w_x = np.zeros((x_features.shape[1]))
        mutual_info = np.zeros((x_features.shape[1]))
        parz_win_width = 1.0 / np.log2(n_trials)
        h_w = -np.sum(prob_w * np.log2(prob_w))

        for i in range(x_features.shape[1]):
            h_w_x[i] = 0
            for j in range(n_classes):
                prob_x_w[j, :, i] = get_pd_pw(x_cls.get(j), i, x_features[:, i])[0]

        t_s = prob_x_w.shape
        n_prob_w_x = np.zeros((n_classes, t_s[1], t_s[2]))
        for i in range(n_classes):
            n_prob_w_x[i, :, :] = prob_x_w[i] * prob_w[i]
        prob_x = np.sum(n_prob_w_x, axis=0)
        # prob_w_x = np.zeros((n_classes, prob_x.shape[0], prob_w.shape[1]))
        for i in range(n_classes):
            prob_w_x[i, :, :] = n_prob_w_x[i, :, :]/prob_x

        for i in range(x_features.shape[1]):
            for j in range(n_trials):
                t_sum = 0.0
                for k in range(n_classes):
                    if prob_w_x[k, j, i] > 0:
                        t_sum += (prob_w_x[k, j, i] * np.log2(prob_w_x[k, j, i]))

                h_w_x[i] -= (t_sum / n_trials)

            mutual_info[i] = h_w - h_w_x[i]

        mifsg = np.asarray(mutual_info)
        return mifsg


    def select_CSP_pairs(self,features_selected,n_pairs):
        features_selected+=1
        sel_groups = np.unique(np.ceil(features_selected/n_pairs))
        paired_features = []
        for i in range(len(sel_groups)):
            for j in range(n_pairs-1,-1,-1):
                paired_features.append(sel_groups[i]*n_pairs-j)

        paired_features = np.asarray(paired_features,dtype=int) - 1

        return paired_features
    
class MLEngine:
    def __init__(self, m_filters=2, feature_selection=True, fs=250):
        self.m_filters = m_filters
        self.feature_selection = feature_selection
        self.fs = fs

    def experiment(self,
                   x_train,
                   y_train,
                   x_test,
                   y_test,
                   ):

        
        fbank = FilterBank(self.fs)
        fbank.get_filter_coeff()
        x_train_fb = fbank.filter_data(x_train, window_details={"tmin":0.5,"tmax":2.5})
        x_test_fb = fbank.filter_data(x_test, window_details={"tmin":0.5,"tmax":2.5})


        y_classes_unique = np.unique(y_train)
        n_classes = len(np.unique(y_train))

        fbcsp = FBCSP(self.m_filters)
        fbcsp.fit(x_train_fb,y_train)
        y_train_predicted = np.zeros((y_train.shape[0], n_classes), dtype=float)
        y_test_predicted = np.zeros((y_test.shape[0], n_classes), dtype=float)

        for j in range(n_classes):
            cls_of_interest = y_classes_unique[j]
            select_class_labels = lambda cls, y_labels: [0 if y == cls else 1 for y in y_labels]

            y_train_cls = np.asarray(select_class_labels(cls_of_interest, y_train))

            x_features_train = fbcsp.transform(x_train_fb,class_idx=cls_of_interest)
            x_features_test = fbcsp.transform(x_test_fb,class_idx=cls_of_interest)

            classifier_type = SVR(gamma='auto')
            classifier = Classifier(classifier_type, self.feature_selection)
            y_train_predicted[:,j] = classifier.fit(x_features_train,np.asarray(y_train_cls,dtype=float))
            y_test_predicted[:,j] = classifier.predict(x_features_test)

        y_train_predicted_multi = self.get_multi_class_regressed(y_train_predicted)
        y_test_predicted_multi = self.get_multi_class_regressed(y_test_predicted)

        tr_acc =np.sum(y_train_predicted_multi == y_train, dtype=float) / len(y_train)
        te_acc =np.sum(y_test_predicted_multi == y_test, dtype=float) / len(y_test)


        print(f'Training Accuracy = {str(tr_acc)}\n')
        print(f'Testing Accuracy = {str(te_acc)}\n \n')
        out_dict = {
            "train_acc": tr_acc,
            "test_acc": te_acc,
            "test_scores": y_test_predicted,
            "train_preds": y_train_predicted_multi,
            "test_preds": y_test_predicted_multi,
        }

        return out_dict
    
    def get_multi_class_regressed(self, y_predicted):
        y_predict_multi = np.asarray([np.argmin(y_predicted[i,:]) for i in range(y_predicted.shape[0])])
        return y_predict_multi




        
    
        