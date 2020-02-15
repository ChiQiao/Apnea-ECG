import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import plotly.graph_objects as go
from hrvanalysis import get_time_domain_features, get_csi_cvi_features
from scipy import interp, signal


def extract_features(data):
    """ Extract features from heart rate data

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
    """
    fs_new = 2.4  # optimized from hyper-parameter tuning
    thres = 0.015  # optimized from hyper-parameter tuning
    df = pd.DataFrame()

    t_hr, hr = data["t"], data["hr"]
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
        fea_dict.update(
            {
                "md_1min": md,
                "min_r_1min": data_1min.min() - md,
                "max_r_1min": data_1min.max() - md,
                "p25_r_1min": np.percentile(data_1min, 0.25) - md,
                "p75_r_1min": np.percentile(data_1min, 0.75) - md,
                "mean_r_1min": data_1min.mean() - md,
                "std_1min": data_1min.std(),
                "acf1_1min": pd.Series(hr_interp_1min).autocorr(12),
                "acf2_1min": pd.Series(hr_interp_1min).autocorr(24),
            }
        )

        # Time-domain features for data_5min
        md = np.median(data_5min)
        fea_dict.update(
            {
                "md_5min": md,
                "min_r_5min": data_5min.min() - md,
                "max_r_5min": data_5min.max() - md,
                "p25_r_5min": np.percentile(data_5min, 0.25) - md,
                "p75_r_5min": np.percentile(data_5min, 0.75) - md,
                "mean_r_5min": data_5min.mean() - md,
                "std_5min": data_5min.std(),
                "acf1_5min": pd.Series(hr_interp_5min).autocorr(12),
                "acf2_5min": pd.Series(hr_interp_5min).autocorr(24),
            }
        )

        # Heart rate variability for data_1min
        nn_intervals = (np.diff(t_hr[idx_1min]) * 1000 * 60).astype(int)  # Unit in ms
        time_domain_features = get_time_domain_features(nn_intervals)
        time_domain_features = {
            f"{key}_1min": value for key, value in time_domain_features.items()
        }
        nonlinear_features = get_csi_cvi_features(nn_intervals)
        nonlinear_features = {
            f"{key}_1min": value for key, value in nonlinear_features.items()
        }
        fea_dict.update(time_domain_features)
        fea_dict.update(nonlinear_features)

        # Heart rate variability for data_5min
        nn_intervals = (np.diff(t_hr[idx_5min]) * 1000 * 60).astype(int)  # Unit in ms
        time_domain_features = get_time_domain_features(nn_intervals)
        time_domain_features = {
            f"{key}_5min": value for key, value in time_domain_features.items()
        }
        nonlinear_features = get_csi_cvi_features(nn_intervals)
        nonlinear_features = {
            f"{key}_5min": value for key, value in nonlinear_features.items()
        }
        fea_dict.update(time_domain_features)
        fea_dict.update(nonlinear_features)

        # Frequency-domain features
        freqs, psd = signal.periodogram(hr_interp_5min, fs=fs_new)
        psd[freqs > 0.1] = 0
        fea_dict.update(
            {
                "peak": psd.max(),
                "f_peak": freqs[np.argmax(psd)],
                "area_total": psd.sum(),
                "area_lf": psd[freqs < thres].sum(),
                "area_hf": psd[freqs > thres].sum(),
                "area_ratio": psd[freqs > thres].sum() / psd[freqs < thres].sum(),
            }
        )

        df = df.append(fea_dict, ignore_index=True)

    df.dropna(inplace=True)
    return df


def extract_cwt(
    file,
    fs_new=1,
    smooth=True,
    cwt_width=100,
    return_segments=False,
    segment_window=3,
    diagPlot=False,
    xlm=[0, 10],
):
    """ Generate Wavelet spectrogram

        Parameters
        ----------
        file: str
            Name of the pkl file in the data/raw folder
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
    """
    with open("../data/raw/" + file + ".pkl", "rb") as f:
        data = pickle.load(f)
        apn = data["apn"]
        group = file[0].upper()

    hr_raw, t_raw = data["hr"], data["t"]

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
        plt.figure(figsize=(20, 10))
        # Time history plot
        # fig.add_subplot(2, 1, 2)
        plt.subplot(212, position=[0.05, 0.05, 0.9, 0.45])
        plt.plot(t, hr)
        plt.xlim(xlm)
        plt.ylabel("Time series", size=30)
        plt.xlabel("Minute", size=30)
        plt.tick_params(labelsize=30)

        # Wavelet plot
        plt.subplot(211, position=[0.05, 0.5, 0.9, 0.45])
        plt.imshow(
            cwt, cmap="gray", aspect="auto", origin="lower", vmin=-2, vmax=2,
        )
        for minute in range(len(apn)):
            sym = "r-" if apn[minute] else "g-"
            plt.plot(
                np.array([minute, minute + 1]) * 60 * fs_new, [0, 0], sym, linewidth=20
            )

        plt.xlim(np.array(xlm) * 60 * fs_new)
        plt.ylabel("Wavelet", size=30)
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
            seg_cwtmatr.append(
                cwt[:, idx_start : int(idx_start + segment_window * 60 * fs_new)]
            )

        cwt = seg_cwtmatr
        apn = apn[2 + half_window : len(apn) - half_window]

    return cwt, apn, group


def smooth_hr(t_hr, hr):
    """ Remove outliers and smooth heart rate data

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
    """
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


def extract_heart_rate(ecg, fs=100):
    """ Calculate heart rate

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
    """
    window_size = 200
    weights = np.ones(window_size) / window_size
    ecg_trend = np.convolve(ecg, weights, mode="same")
    ecg_detrend = ecg - ecg_trend
    r_idx = extract_r(ecg_detrend, fs)
    t = r_idx[1:] / fs
    hr = 1 / (np.diff(r_idx) / fs)
    return t, hr


def extract_r(ecg_detrend, fs=100):
    """ Extract R peaks from ECG data 
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
    """
    import biosppy.signals.ecg as ECG

    r_idx = list(ECG.christov_segmenter(ecg_detrend, fs))[0]  # Works fine in most cases
    # Assuming HR = 1 bps, check if half of the R peaks are detected
    if len(r_idx) < (len(ecg_detrend) / fs) / 2:
        r_idx_2 = list(ECG.hamilton_segmenter(ecg_detrend, fs))[
            0
        ]  # Might return negative R peaks
        if len(r_idx_2) > len(r_idx):
            r_idx = r_idx_2
    return r_idx


def extract_pqrst(ecg_detrend, r_idx, fs=100, diagPlot=False):
    """ Extract peaks from ECG data
        Based on results from extract_r, extract PQRST peaks

        Returns
        -------
        peak_idx, peak_val: N x 5 numpy array, index and value of peaks of PQRST
    """
    p_idx, q_idx, s_idx, t_idx = [], [], [], []
    for i in range(len(r_idx) - 1):
        idx_1 = int(r_idx[i] + (r_idx[i + 1] - r_idx[i]) * 0.3)
        idx_2 = int(r_idx[i] + (r_idx[i + 1] - r_idx[i]) * 0.5)
        idx_3 = int(r_idx[i] + (r_idx[i + 1] - r_idx[i]) * 0.6)
        idx_4 = int(r_idx[i] + (r_idx[i + 1] - r_idx[i]) * 0.9)

        try:
            idx = np.argmin(ecg_detrend[r_idx[i] : idx_1])
        except:
            idx = 0
        s_idx_ = r_idx[i] + idx
        s_idx.append(s_idx_)

        try:
            idx = np.argmax(ecg_detrend[s_idx_:idx_2])
        except:
            idx = 0
        t_idx_ = s_idx_ + idx
        t_idx.append(t_idx_)

        try:
            idx = np.argmin(ecg_detrend[idx_4 : r_idx[i + 1]])
        except:
            idx = 0
        q_idx_ = idx_4 + idx
        q_idx.append(q_idx_)

        try:
            idx = np.argmax(ecg_detrend[idx_3:q_idx_])
        except:
            idx = 0
        p_idx_ = idx_3 + idx
        p_idx.append(p_idx_)

    # Make pairs
    p_idx = p_idx[:-1]
    q_idx = q_idx[:-1]
    r_idx = r_idx[1:-1]
    s_idx = s_idx[1:]
    t_idx = t_idx[1:]

    # Form index result (row: observation, column: each pulse)
    res = np.transpose(np.vstack((p_idx, q_idx, r_idx, s_idx, t_idx)))
    # res = res[np.logical_not(np.isnan(res.sum(axis=1))), :].astype(int) # Delete rows with nan

    if diagPlot:
        t = np.arange(len(ecg_detrend)) / fs
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=t, y=ecg_detrend, mode="lines", name="ECG_raw"))
        fig.add_trace(
            go.Scatter(
                x=t[res[:, 0]], y=ecg_detrend[res[:, 0]], mode="markers", name="P"
            )
        )
        fig.add_trace(
            go.Scatter(
                x=t[res[:, 1]], y=ecg_detrend[res[:, 1]], mode="markers", name="Q"
            )
        )
        fig.add_trace(
            go.Scatter(
                x=t[res[:, 2]], y=ecg_detrend[res[:, 2]], mode="markers", name="R"
            )
        )
        fig.add_trace(
            go.Scatter(
                x=t[res[:, 3]], y=ecg_detrend[res[:, 3]], mode="markers", name="S"
            )
        )
        fig.add_trace(
            go.Scatter(
                x=t[res[:, 4]], y=ecg_detrend[res[:, 4]], mode="markers", name="T"
            )
        )
        fig.show()

    return res, ecg_detrend[res]
