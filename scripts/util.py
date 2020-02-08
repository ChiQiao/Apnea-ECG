import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import plotly.graph_objects as go
from hrvanalysis import get_time_domain_features, get_csi_cvi_features
from scipy import interp, signal
from sklearn import preprocessing
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

def extract_features(data):
    ''' Extract features from heart rate data

        Parameters
        ----------
        data: dict with key t and hr
            data['t'] contains a numpy array indicating time in minutes
            data['hr'] contains a numpy array with the same size of data['t'] 
                indicating heart rate in beats per second
        
        Returns
        -------
        df: pandas DataFrame
            Features in time and frequency domains for each minute
    '''
    fs_new = 2.4 # optimized from hyper-parameter tuning
    thres = 0.015 # optimized from hyper-parameter tuning
    df = pd.DataFrame()

    t_hr, hr = data['t'], data['hr']
    t_hr, hr_smth = smooth_hr(t_hr, hr)
    total_minute = int(t_hr[-1] - t_hr[0])
    
    # Resample data for frequency-domain analysis
    t_interp = np.arange(t_hr[0], t_hr[-1], 1 / fs_new / 60)
    hr_interp = np.interp(t_interp, t_hr, hr_smth)

    # Extract features from each segment
    for minute in range(total_minute - 4):
        fea_dict = {}
        idx_1min = (t_hr > minute + 2) & (t_hr < minute + 3)
        idx_5min = (t_hr > minute) & (t_hr < minute + 5)
        data_1min, data_5min = hr_smth[idx_1min], hr_smth[idx_5min]

        hr_interp_1min = hr_interp[(t_interp > minute + 2) & (t_interp < minute + 3)]
        hr_interp_5min = hr_interp[(t_interp > minute) & (t_interp < minute + 5)]

        # Discard segment if less than 30 heart beats detected
        if len(data_1min) < 30: 
            continue

        # Time-domain features for data_1min
        md = np.median(data_1min)
        fea_dict.update({
            'md_1min': md,
            'min_r_1min': data_1min.min() - md,
            'max_r_1min': data_1min.max() - md,
            'p25_r_1min': np.percentile(data_1min, 0.25) - md,
            'p75_r_1min': np.percentile(data_1min, 0.75) - md,
            'mean_r_1min': data_1min.mean() - md,
            'std_1min': data_1min.std(),
            'acf1_1min': pd.Series(hr_interp_1min).autocorr(12),
            'acf2_1min': pd.Series(hr_interp_1min).autocorr(24),
        })

        # Time-domain features for data_5min
        md = np.median(data_5min)
        fea_dict.update({
            'md_5min': md,
            'min_r_5min': data_5min.min() - md,
            'max_r_5min': data_5min.max() - md,
            'p25_r_5min': np.percentile(data_5min, 0.25) - md,
            'p75_r_5min': np.percentile(data_5min, 0.75) - md,
            'mean_r_5min': data_5min.mean() - md,
            'std_5min': data_5min.std(),
            'acf1_5min': pd.Series(hr_interp_5min).autocorr(12),
            'acf2_5min': pd.Series(hr_interp_5min).autocorr(24),
        })

        # Heart rate variability for data_1min
        nn_intervals = (np.diff(t_hr[idx_1min]) * 1000 * 60).astype(int) # Unit in ms
        time_domain_features = get_time_domain_features(nn_intervals)
        time_domain_features = {f'{key}_1min': value for key, value in time_domain_features.items()}
        nonlinear_features = get_csi_cvi_features(nn_intervals)
        nonlinear_features = {f'{key}_1min': value for key, value in nonlinear_features.items()}
        fea_dict.update(time_domain_features)
        fea_dict.update(nonlinear_features)

        # Heart rate variability for data_5min
        nn_intervals = (np.diff(t_hr[idx_5min]) * 1000 * 60).astype(int) # Unit in ms
        time_domain_features = get_time_domain_features(nn_intervals)
        time_domain_features = {f'{key}_5min': value for key, value in time_domain_features.items()}
        nonlinear_features = get_csi_cvi_features(nn_intervals)
        nonlinear_features = {f'{key}_5min': value for key, value in nonlinear_features.items()}
        fea_dict.update(time_domain_features)
        fea_dict.update(nonlinear_features)

        # Frequency-domain features
        freqs, psd = signal.periodogram(hr_interp_5min, fs=fs_new)
        psd[freqs > 0.1] = 0
        fea_dict.update({
            'peak': psd.max(),
            'f_peak': freqs[np.argmax(psd)],
            'area_total': psd.sum(),
            'area_lf': psd[freqs < thres].sum(),
            'area_hf': psd[freqs > thres].sum(),
            'area_ratio': psd[freqs > thres].sum() / psd[freqs < thres].sum(),
        })

        df = df.append(fea_dict, ignore_index=True)

    df.dropna(inplace=True)
    return df


def smooth_hr(t_hr, hr):
    ''' Remove outliers and smooth heart rate data

        Parameters
        ----------
        t_hr: numpy array
            Time of heart rate data in minutes, same size as hr
        hr: numpy array
            Heart rate data in beats per second

        Returns
        -------
        t_hr: numpy array
            A subset of the input t_hr with outliers removed
        hr_smth: numpy array
            Smoothed heart rate with the same size of t_hr
    '''
    b, a = signal.butter(3, 0.1)

    # Remove outliers 
    idx_valid = (hr < 2) & (hr > 0.5)
    hr, t_hr = hr[idx_valid], t_hr[idx_valid]
    
    # Filter out high-frequency noise
    hr_smth = signal.filtfilt(b, a, hr)
    
    # Remove outliers again
    idx_valid = (hr < hr_smth + 0.2) & (hr > hr_smth - 0.2)
    hr, t_hr = hr[idx_valid], t_hr[idx_valid]
    
    # Filter out high-frequency noise
    hr_smth = signal.filtfilt(b, a, hr)

    return t_hr, hr_smth

def get_cwt(file, fs_new=1, smooth=True, cwt_width=100, return_segments=False, segment_window=3, diagPlot=False, xlm=[0, 10]):
    ''' Generate Wavelet spectrogram

        Parameters
        ----------
        file: str
            Name of the pkl file in the HR_data folder
        fs_new: double
            Resampling frequency
        smooth: boolean
            Whether to smooth data before wavelet transformation
        cwt_width: int
            Width parameter for the scipy.signal.cwt function
        return_segments: boolean
            Whether to return cwt and apn in segments with window size determined by segment_window
        segment_window: int
            Window size in minutes to associate with labels
            Odd number required
            Only used when return_segments=True
        diagPlot: boolean
            Whether to generate diagnostic plots
        xlm: list
            X axis limit for diagnostic plot in minutes, [x_lowerbound, x_upperbound]
            Only used when diagPlot=True

        Returns
        -------
        cwt: numpy 2D array if return_segments=False, list of numpy 2D array if return_segments=True
            Wavelet transformation
        apn: numpy array
            Apnea label for each minute
        group: str
            Severity group of the file (determined directly from file name)
    '''
    with open('../HR_data/' + file + '.pkl', 'rb') as f:
        data = pickle.load(f)
        apn = data['apn']
        group = file[0].upper() 

    hr_raw, t_raw = data['hr'], data['t']

    # Smooth data
    if smooth:
        t_raw, hr_raw = smooth_hr(t_raw, hr_raw)

    # Resample data for frequency-domain analysis
    t = np.arange(t_raw[0], t_raw[-1], 1 / fs_new / 60)
    hr = np.interp(t, t_raw, hr_raw)

    widths = np.arange(1, cwt_width + 1)
    cwt = signal.cwt(hr, signal.ricker, widths)
    
    # Diagnose plot
    if diagPlot:
        plt.figure(figsize=(20,10))
        # Time history plot
        # fig.add_subplot(2, 1, 2)
        plt.subplot(212, position=[0.05, 0.05, 0.9, 0.45])
        plt.plot(t, hr)
        plt.xlim(xlm)
        plt.ylabel('Time series', size=30)
        plt.xlabel('Minute', size=30)
        plt.tick_params(labelsize=30)

        # Wavelet plot
        plt.subplot(211, position=[0.05, 0.5, 0.9, 0.45])
        plt.imshow(cwt, cmap='gray', aspect='auto', origin='lower', vmin=-2, vmax=2,)
        for minute in range(len(apn)):
            sym = 'r-' if apn[minute] else 'g-' 
            plt.plot(np.array([minute, minute+1]) * 60 * fs_new, [0, 0], sym, linewidth=20) 

        plt.xlim(np.array(xlm) * 60 * fs_new)
        plt.ylabel('Wavelet', size=30)
        plt.xticks([])
        plt.tick_params(labelsize=30)
        plt.show()
    
    if return_segments:
        half_window = int(segment_window / 2)
        seg_cwtmatr = []
        # Skip first and last 2-minute data to eliminate boundary effects of CWT
        for idx in range(2 + half_window, len(apn) - half_window):
            minute_start = idx - half_window
            idx_start = int(minute_start * 60 * fs_new)
            seg_cwtmatr.append(cwt[:, idx_start : int(idx_start + segment_window * 60 * fs_new)])
        
        cwt = seg_cwtmatr
        apn = apn[2 + half_window : len(apn) - half_window]
        
    return cwt, apn, group


def feature_select(mdl, df, train_df, feature_col, n=4):
    ''' Select features iteratively
        Ranke feature importance, and test cross validation result by eliminating the least important feature
        Stops when accuracy no longer improves after trying each feature

        Parameters
        ----------
        mdl: sklearn model
        df: DataFrame
            Column 'apn' as target
        train_df: DataFrame
            For stratified train-test split
            Contains column 'file' and 'group'
        features_col: list
            Features used for model training
        n: int
            K-fold cross validation

        Returns
        -------
        features: list
            Selected features
    '''
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


def model_evaluation_CV(mdl, df, file_df, feature_col, n=4, normalize=True, plot_roc=False):
    ''' Evaluate model accuracy using Stratified K-fold CV
        Split training dataset according to patient group

        Parameters
        ----------
        mdl: sklearn model
        df: DataFrame
            Column 'apn' as target
        file_df: DataFrame
            For stratified cross validation
            Contains column 'file' and 'group'
        feature_col: list
            Features for model training
        n: int
            K-fold cross validation
        normalize: boolean
            Whether to standardize features during model training
        plot_roc: boolean
            Whetehr to generate ROC curve for each fold

        Returns
        -------
        res: dict includes the following keys
            minute_auc_mean: Mean AUC of minute-wise prediction
            minute_auc_cv: Detailed AUC of minute-wise prediction for each fold
            group_auc: Macro mean of group-wise AUC (for group A & C only)
            group_f1_best: highest macro F1 score (group A, B, and C) for group-wise prediction 
            thres_best: Corresponding threshold for group_f1_best
            minute_detail: dict recording minitue-wise prediction for each patient
            group_detail: DataFrame recording group-wise prediction
            multiclass_auc: dict recording TPR and FPR of group-wise prediction
        When plot_roc=True, additional values are included:
            mean_fpr_minute: Mean false positive rate for minute-wise prediction
            mean_tpr_minute: Mean true positive rate for minute-wise prediction
            mean_auc_minute: AUC from mean_fpr_minute and mean_tpr_minute
    '''
    # Initialize
    auc_val = []
    group_res = file_df.copy(deep=True).set_index('file')
    group_res.rename(columns={'group': 'true'}, inplace=True)
    minute_res = {}
    if plot_roc:
        tprs, aucs = [], []
        mean_fpr = np.linspace(0, 1, 100)
        _, ax = plt.subplots()

    skf = StratifiedKFold(n_splits=n)
    for idx_train, idx_val in skf.split(file_df, file_df['group']):
        file_train, file_val = file_df.loc[idx_train, 'file'], file_df.loc[idx_val, 'file']
        X_train, y_train = df.loc[df.file.isin(file_train), feature_col], df.loc[df.file.isin(file_train), 'apn']
        X_val, y_val = df.loc[df.file.isin(file_val), feature_col], df.loc[df.file.isin(file_val), 'apn']

        if normalize:
            scaler = preprocessing.StandardScaler().fit(X_train)
            X_train = scaler.transform(X_train)
            X_val = scaler.transform(X_val)

        mdl.fit(X_train, y_train)
        auc_val.append(metrics.roc_auc_score(y_val, mdl.predict_proba(X_val)[:, 1]))

        if plot_roc:
            viz = metrics.plot_roc_curve(mdl, X_val, y_val, alpha=0.3, lw=1, ax=ax)
            interp_tpr = interp(mean_fpr, viz.fpr, viz.tpr)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)
            aucs.append(viz.roc_auc)

        # Accuracy for group diagnosis
        for file in file_val:
            X_val, y_val = df.loc[df.file.isin([file]), feature_col], df.loc[df.file.isin([file]), 'apn']
            X_val = scaler.transform(X_val) if normalize else X_val
            # Group prediction
            y_pred = mdl.predict(X_val)
            group_res.loc[file,'pred'] = ecg_diagnose(y_pred)
            group_res.loc[file,'true'] = ecg_diagnose(y_val.values) # Original group might be wrong (a10 is identified as B)
            # Minute-wise prediction probability
            y_pred_prob = mdl.predict_proba(X_val)
            minute_res[file] = np.vstack((y_val, y_pred_prob[:, 1]))

    minute_auc = np.mean(auc_val)
    group_auc_macro, group_f1_best, thres_best, multiclass_auc = eval_multiclass_auc(group_res, minute_res)
    
    res = {
        'minute_auc_mean': minute_auc, 
        'minute_auc_cv': auc_val,
        'group_auc': group_auc_macro, 
        'group_f1_best': group_f1_best, 
        'thres_best': thres_best, 
        'minute_detail': minute_res,
        'group_detail': group_res,
        'multiclass_auc': multiclass_auc,
    }

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

        res['mean_fpr_minute'] = mean_fpr
        res['mean_tpr_minute'] = mean_tpr
        res['mean_auc_minute'] = mean_auc

    return res


def model_evaluation_test(mdl, df, file_df, feature_col, scaler, thres):
    ''' Evaluate model accuracy on testing data
        
        Parameters
        ----------
        mdl: sklearn model
            Trained model
        df: DataFrame
            Features of the testing dataset
            Column 'apn' as target
        file_df: DataFrame
            For recording group-wise prediction
        feature_col: list
            Features for model prediction
        scaler: sklearn scaler
            For standardize features
        thres: double
            Between 0 and 1, threshold for minute-wise prediction

        Returns
        -------
        res: dict includes the following keys
            group_res: DataFrame recording group-wise prediction
            minute_auc: AUC of the minute-wise prediction
            minute_detail: dict recording minitue-wise prediction for each patient
            group_detail: DataFrame recording group-wise prediction
    '''
    # Evaluate model accuracy using Stratified K-fold CV
    # Note: Stratification is based on patient group (A, B, C), and then samples are formed
    
    # Initialize
    group_res = file_df.copy(deep=True).set_index('file')
    group_res.rename(columns={'group': 'true'}, inplace=True)
    minute_res = {}

    # Normalize df
    df_norm = pd.DataFrame(scaler.transform(df[feature_col]), columns=feature_col)
    df = df_norm.join(df[['apn', 'file', 'group']])

    # Overall AUC
    minute_auc = metrics.roc_auc_score(df['apn'], mdl.predict_proba(df[feature_col])[:, 1])

    # Accuracy for group diagnosis
    for file in file_df['file']:
        X, y = df.loc[df.file.isin([file]), feature_col], df.loc[df.file.isin([file]), 'apn']
        # Group prediction
        y_pred = (mdl.predict_proba(X)[:, 1] > thres).astype(int)
        group_res.loc[file,'pred'] = ecg_diagnose(y_pred)
        group_res.loc[file,'true'] = ecg_diagnose(y.values) # Original group might be wrong (a10 is identified as B)
        minute_res[file] = np.vstack((y, y_pred))

    res_detail = {
        'group_res': group_res,
        'minute_auc': minute_auc, 
        'minute_detail': minute_res,
        'group_detail': group_res,
    }

    return res_detail


def eval_multiclass_auc(group_res, minute_res):
    ''' Calculate multiclass ROC

        Parameters
        ----------
        group_res: DataFrame
            Group-wise prediction result
        minute_res: dict
            Minute-wise prediction for each patient
            Value of each key is a numpy array, with the first row as the true label, 
            and the second row as the prediction probability

        Returns
        -------
        auc_macro: scalar
            Macro mean of AUC from the ROC curve of group A & C
        f1_macro_opt: scalar
            Optimal macro averaged F1 score for group A, B and C
        thres_opt: scalar
            Corresponding threshold for f1_macro_opt
        multiclass_auc: dict 
            Detailed ROC curves for each group and macro F1
    '''
    multiclass_auc = {
        'fpr_A': [], 'tpr_A': [], 
        'fpr_B': [], 'tpr_B': [], 
        'fpr_C': [], 'tpr_C': [],
        'f1_macro': [],
    }
    group_true = group_res['true'].values
    
    # AUC
    thres_all = np.linspace(0, 1, 101)
    for thres in thres_all:
        group_pred = np.array([ecg_diagnose(minute_res[patient][1, :] > thres) for patient in group_res.index])
        multiclass_auc['f1_macro'].append(metrics.f1_score(group_true, group_pred, average='macro'))
        for group in list('ABC'):
            temp_true = group_true == group
            temp_pred = group_pred == group

            # calculate tpr & fpr
            tn, fp, fn, tp = metrics.confusion_matrix(temp_true, temp_pred).ravel()
            fpr = fp / (tn + fp)
            tpr = tp / (tp + fn)

            multiclass_auc['fpr_' + group].append(fpr)
            multiclass_auc['tpr_' + group].append(tpr)
            
    # Macro avg. of AUC for class A & C (B does not form intact ROC curve)
    f1_macro_opt = np.max(multiclass_auc['f1_macro'])
    thres_opt = thres_all[np.argmax(multiclass_auc['f1_macro'])]
    auc_macro = np.mean([
        metrics.auc(multiclass_auc['fpr_A'], multiclass_auc['tpr_A']),
        metrics.auc(multiclass_auc['fpr_C'], multiclass_auc['tpr_C'])
    ])
    
    return auc_macro, f1_macro_opt, thres_opt, multiclass_auc    


def ecg_diagnose(apn):
    ''' Diagnose severity of sleep apnea

        Parameters
        ----------
        apn: list
            Apnea label for each minute

        Returns
        -------
        Severity group
    '''
    # Total minutes of apnea
    apnea_total = sum(apn)

    # Maximum hourly Apnea Index
    total_hour = int(len(apn) / 60)
    y_pred_hourly = np.reshape(apn[ : total_hour * 60], (total_hour, 60))
    AI_hourly = y_pred_hourly.sum(axis=1)
    # If data in the last hour exceed 30 minutes, then convert to hourly result
    y_pred_left = apn[total_hour * 60 : ]
    # if len(y_pred_left) >= 30:
    #     AI_hourly = np.append(AI_hourly, sum(y_pred_left) * 60 / len(y_pred_left))
    #     total_hour += 1
    AI_max = AI_hourly.max()

    if AI_max >= 10 and apnea_total >= 100:
        return 'A'
    elif AI_max >= 5 and apnea_total >= 5:
        return 'B'
    else:
        return 'C'


def __detrend_data(data, window_size=200):
    weights = np.ones(window_size) / window_size
    ecg_trend = np.convolve(data, weights, mode='same')
    return data - ecg_trend


def get_normal_segment_idx(ecg, ratio_lb, ratio_ub, diagPlot=False):
    ''' Detect abnormal ecg recording segments

        Parameters
        ----------
        ecg: numpy 2D array
            ECG recording for each minute
        ratio_lb: scalar
            Lower bound of segment standard deviation normalized by overall median
        ratio_ub: scalar
            Upper bound of segment standard deviation normalized by overall median
        diagPlot: boolean
            Whether to generate diagnostic plot

        Returns
        -------
        idx_valid: numpy array of boolean
            Indicates valid rows of ecg
    '''
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


def get_heart_rate(ecg, fs=100):
    ''' Calculate heart rate

        Parameters
        ----------
        ecg: numpy 2D array
        fs: scalar, sampling frequency

        Returns
        -------
        t: numpy array
            Time of heart rate
        hr: numpy array
            Heart rate in beats per second, with the same size as t
    '''
    ecg_detrend = __detrend_data(ecg)
    r_idx = extract_r(ecg_detrend, fs)
    t = r_idx[1: ] / fs
    hr = 1 / (np.diff(r_idx) / fs)
    return t, hr


def extract_r(ecg_detrend, fs=100):
    ''' Extract R peaks from ECG data 
        ECG data should have the length around 1 minute to get a stable result

        Parameters
        ----------
        ecg_detrend: numpy array or list
            Detrended ECG data 
        fs: scalar
            Sampling frequency of ecg_detrend

        Returns
        -------
        r_idx: Index of R peaks for ecg_detrend
    '''
    import biosppy.signals.ecg as ECG
    r_idx = list(ECG.christov_segmenter(ecg_detrend, fs))[0] # Works fine in most cases
    # Assuming HR = 1 bps, check if half of the R peaks are detected
    if len(r_idx) < (len(ecg_detrend) / fs) / 2:
        r_idx_2 = list(ECG.hamilton_segmenter(ecg_detrend, fs))[0] # Might return negative R peaks
        if len(r_idx_2) > len(r_idx):
            r_idx = r_idx_2
    return r_idx


def extract_pqrst(ecg_detrend, r_idx, fs=100, diagPlot=False):
    ''' Extract peaks from ECG data
        Based on results from extract_r, extract PQRST peaks

        Returns
        -------
        peak_idx, peak_val: N x 5 numpy array, index and value of peaks of PQRST
    '''
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


