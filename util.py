import numpy as np
from scipy import interp
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import biosppy.signals.ecg as ECG
from sklearn import preprocessing
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn import metrics


fs = 100

def feature_select(mdl, df, train_df, feature_col, n=4):
    # Baseline accuracy
    _, acc_base, _ = model_evaluation_CV(clone(mdl), df, train_df, feature_col, n=n)
    print(f'Baseline accuracy: {acc_base:.3f}')

    # Rank features in descending orders 
    selector = RFE(LogisticRegression(solver='lbfgs', max_iter=1e6))
    selector = selector.fit(
        X=(df[feature_col] - df[feature_col].mean()) / df[feature_col].std(), 
        y=df['apn'])
    features = [feature for _, feature in sorted(zip(selector.ranking_, df[feature_col].columns), reverse=True)]

    # New accuracy using RFE
    _, acc_rfe, _ = model_evaluation_CV(clone(mdl), df, train_df, feature_col[selector.support_], n=n)
    print(f'Accuracy using RFE: {acc_rfe:.3f}')

    # Search for optimal features that improve model performance
    _, acc_base, _ = model_evaluation_CV(clone(mdl), df, train_df, features, n=n)
    acc_new = acc_base
    print('**********************************')
    print(f'Baseline accuracy: {acc_base:.3f}')
    while True:
        acc_base = acc_new
        for idx in range(len(features)):
            _, acc_, _ = model_evaluation_CV(clone(mdl), df, train_df, np.delete(features, idx), n=n)
            if acc_ > acc_base:
                print(f'Accuracy revoming {features[idx]}: {acc_:.3f}')
                features = np.delete(features, idx)
                acc_new = acc_
                break
                
        if acc_new == acc_base:
            break

    return features


def model_evaluation_CV(mdl, df, train_df, feature_col, n=5,
    normalize=True, detail_pred=False, plot_roc=False):
    # Evaluate model accuracy using Stratified K-fold CV
    # Note: Stratification is based on patient group (A, B, C), and then samples are formed
    skf = StratifiedKFold(n_splits=n)
    acc_train, acc_val = [], []

    if detail_pred:
        patient_pred = train_df.copy(deep=True).set_index('file')
        patient_pred.rename(columns={'group': 'true'}, inplace=True)
        seg_pred = {}

    if plot_roc:
        tprs, aucs = [], []
        mean_fpr = np.linspace(0, 1, 100)
        _, ax = plt.subplots()
    
    for idx_train, idx_val in skf.split(train_df, train_df['group']):
        file_train, file_val = train_df.loc[idx_train, 'file'], train_df.loc[idx_val, 'file']
        X_train, y_train = df.loc[df.file.isin(file_train), feature_col], df.loc[df.file.isin(file_train), 'apn']
        X_val, y_val = df.loc[df.file.isin(file_val), feature_col], df.loc[df.file.isin(file_val), 'apn']
        
        if normalize:
            scaler = preprocessing.StandardScaler().fit(X_train)
            X_train = scaler.transform(X_train)
            X_val = scaler.transform(X_val)

        mdl.fit(X_train, y_train)
        acc_train.append(metrics.accuracy_score(y_train, mdl.predict(X_train)))
        acc_val.append(metrics.accuracy_score(y_val, mdl.predict(X_val)))
        
        if plot_roc:
            viz = metrics.plot_roc_curve(mdl, X_val, y_val,
                                alpha=0.3, lw=1, ax=ax)
            interp_tpr = interp(mean_fpr, viz.fpr, viz.tpr)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)
            aucs.append(viz.roc_auc)
            
        # Accuracy for group diagnosis
        if detail_pred:
            for file in file_val:
                X_val, y_val = df.loc[df.file.isin([file]), feature_col], df.loc[df.file.isin([file]), 'apn']
                X_val = scaler.transform(X_val) if normalize else X_val
                y_pred = mdl.predict(X_val)
                y_pred_prob = mdl.predict_proba(X_val)
                patient_pred.loc[file,'pred']=ecg_diagnose(y_pred)
                seg_pred[file] = np.vstack((y_val, y_pred, y_pred_prob[:, 1]))
    
    if plot_roc:        
        ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
                label='Chance', alpha=.8)

        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = metrics.auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)
        ax.plot(mean_fpr, mean_tpr, color='b',
                label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
                lw=2, alpha=.8)

        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                        label=r'$\pm$ 1 std. dev.')

        ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05])
        ax.legend(loc="lower right")
        plt.show()

    res_detail = {}
    if detail_pred:
        res_detail['patient_pred'] = patient_pred
        res_detail['seg_pred'] = seg_pred
    if plot_roc:
        res_detail['mean_fpr'] = mean_fpr
        res_detail['mean_tpr'] = mean_tpr
        res_detail['mean_auc'] = mean_auc
    
    return np.mean(acc_train), np.mean(acc_val), res_detail


def train_test_sample_split(df, X_label, y_label):
    file_train = pd.read_csv('resources/File_train.csv')
    file_test = pd.read_csv('resources/File_test.csv')
    file_samp = pd.read_csv('resources/File_sample.csv')

    X_train = df[df.file.isin(file_train.file)][X_label]
    y_train = df[df.file.isin(file_train.file)][y_label]
    X_test = df[df.file.isin(file_test.file)][X_label]
    y_test = df[df.file.isin(file_test.file)][y_label]

    res = {}
    res['X_A'] = df[df.file.isin(file_samp[file_samp.group.isin(['A'])].file)][X_label]
    res['X_B'] = df[df.file.isin(file_samp[file_samp.group.isin(['B'])].file)][X_label]
    res['X_C'] = df[df.file.isin(file_samp[file_samp.group.isin(['C'])].file)][X_label]

    return X_train, X_test, y_train, y_test, res

def ecg_diagnose(apn):
    # Total minutes of apnea
    apnea_total = sum(apn)

    # Maximum hourly Apnea Index
    total_hour = int(len(apn) / 60)
    y_pred_hourly = np.reshape(apn[ : total_hour * 60], (total_hour, 60))
    AI_hourly = y_pred_hourly.sum(axis=1)
    # If data in the last hour exceed 30 minutes, then convert to hourly result
    y_pred_left = apn[total_hour * 60 : ]
    if len(y_pred_left) >= 30:
        AI_hourly = np.append(AI_hourly, sum(y_pred_left) * 60 / len(y_pred_left))
        total_hour += 1
    AI_max = AI_hourly.max()

    if AI_max >= 10 and apnea_total >= 100:
        return 'A'
    elif AI_max >= 5 and apnea_total >= 5:
        return 'B'
    else:
        return 'C'


def get_nonoutlier_idx(data, m=2.):
    # Find non-outliers
    #   idx_valid = get_nonoutlier_idx(data, m)
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d / (mdev if mdev else 1.)
    idx_valid = s < m
    return idx_valid


def detrend_data(data, window_size=200):
    weights = np.ones(window_size) / window_size
    ecg_trend = np.convolve(data, weights, mode='same')
    return data - ecg_trend


def get_normal_segment_idx(ecg, ratio_lb, ratio_ub, diagPlot):
    ecg_sd = ecg.std(axis=1)
    ecg_sd_med = np.median(ecg_sd)
    idx_valid = (ecg_sd < ecg_sd_med * ratio_ub) & (ecg_sd > ecg_sd_med * ratio_lb)

    if diagPlot:
        plt.figure()
        for i in range(ecg.shape[0]):
            if idx_valid[i] == 0:
                plt.plot(np.arange(6000) + 6000 * i, ecg[i, :], 'r-')
            else:
                plt.plot(np.arange(6000) + 6000 * i, ecg[i, :], 'b-')
        # plt.show()

    return idx_valid



def get_heart_rate(ecg):
    # Calculate heart rate
    #   t, hr = get_heart_rate(ecg)
    # Output
    #   t: Time in second
    #   hr: Heart rate corresponding to t, unit in bps

    ecg_detrend = detrend_data(ecg)
    r_idx = extract_r(ecg_detrend, fs)
    t = r_idx[1: ] / fs
    hr = 1 / (np.diff(r_idx) / fs)
    return t, hr


def extract_r(ecg_detrend, fs):
    r_idx = list(ECG.christov_segmenter(ecg_detrend, fs))[0] # Works fine in most cases
    # Assuming HR = 1 bps, check if half of the R peaks are detected
    if len(r_idx) < (len(ecg_detrend) / fs) / 2:
        r_idx_2 = list(ECG.hamilton_segmenter(ecg_detrend, fs))[0] # Might return negative R peaks
        if len(r_idx_2) > len(r_idx):
            r_idx = r_idx_2
    return r_idx


def extract_pqrst(ecg_detrend, r_idx, fs, diagPlot):
    # Extract peaks from ECG data
    #   peak_idx, peak_val = extract_pqrst(ecg_data, fs, diagPlot)
    #   peak_idx, peak_val: N x 5 np array, index and value of peaks of PQRST

    p_idx, q_idx, s_idx, t_idx = [], [], [], []
    for i in range(len(r_idx)-1):
        idx_1 = int(r_idx[i] + (r_idx[i+1] - r_idx[i]) * 0.3)
        idx_2 = int(r_idx[i] + (r_idx[i+1] - r_idx[i]) * 0.5)
        idx_3 = int(r_idx[i] + (r_idx[i+1] - r_idx[i]) * 0.6)
        idx_4 = int(r_idx[i] + (r_idx[i+1] - r_idx[i]) * 0.9)

        try:
            idx = np.argmin(ecg_detrend[r_idx[i] : idx_1])
        except:
            idx = 0
        s_idx_ = r_idx[i] + idx
        s_idx.append(s_idx_)

        try:
            idx = np.argmax(ecg_detrend[s_idx_ : idx_2])
        except:
            idx = 0
        t_idx_ = s_idx_ + idx
        t_idx.append(t_idx_)

        try:
            idx = np.argmin(ecg_detrend[idx_4 : r_idx[i+1]])
        except:
            idx = 0
        q_idx_ = idx_4 + idx
        q_idx.append(q_idx_)

        try:
            idx = np.argmax(ecg_detrend[idx_3 : q_idx_])
        except:
            idx = 0
        p_idx_ = idx_3 + idx
        p_idx.append(p_idx_)

    # Make pairs
    p_idx = p_idx[ : -1]
    q_idx = q_idx[ : -1]
    r_idx = r_idx[1 : -1]
    s_idx = s_idx[1 : ]
    t_idx = t_idx[1 : ]

    # Form index result (row: observation, column: each pulse)
    res = np.transpose(np.vstack(
        (p_idx, q_idx, r_idx, s_idx, t_idx)
    ))
    # res = res[np.logical_not(np.isnan(res.sum(axis=1))), :].astype(int) # Delete rows with nan

    if diagPlot:
        t = np.arange(len(ecg_detrend)) / fs
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=t, y=ecg_detrend, mode='lines', name='ECG_raw'))
        fig.add_trace(go.Scatter(
            x=t[res[:, 0]], y=ecg_detrend[res[:, 0]], mode='markers', name='P'))
        fig.add_trace(go.Scatter(
            x=t[res[:, 1]], y=ecg_detrend[res[:, 1]], mode='markers', name='Q'))
        fig.add_trace(go.Scatter(
            x=t[res[:, 2]], y=ecg_detrend[res[:, 2]], mode='markers', name='R'))
        fig.add_trace(go.Scatter(
            x=t[res[:, 3]], y=ecg_detrend[res[:, 3]], mode='markers', name='S'))
        fig.add_trace(go.Scatter(
            x=t[res[:, 4]], y=ecg_detrend[res[:, 4]], mode='markers', name='T'))
        fig.show()

    return res, ecg_detrend[res]


