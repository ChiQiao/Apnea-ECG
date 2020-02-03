import wfdb
import numpy as np

def single_file_res(data_folder, file, apn, t_end):
    apn = apn[1: -1]  # Remove first and last APN label

    # ECG data (ecg)
    ecg, _ = get_ecg(data_folder + file)
    ecg = np.array(ecg).flatten()
    assert len(ecg) >= t_end * 100, 'Period of ecg shorter than label'
    ecg = ecg[: int(t_end * 100)]  # Discard the tail
    # Discard first 30 sec and last 30 sec, so apn is at middle
    ecg = ecg[3000: -3000].reshape(len(apn), 6000)

    # QRS label
    qrs, _, t_qrs = get_qrs(data_folder + file)
    qrs = np.array(qrs)
    qrs = qrs[t_qrs <= t_end]
    t_qrs = t_qrs[t_qrs <= t_end]
    idx_qrs = (t_qrs * 100).astype(int)

    # R peak mask (r_peaks)
    r_peaks = np.zeros(int(t_end*100))
    r_peaks[idx_qrs[qrs == 'N']] = 1
    r_peaks = r_peaks[3000: -3000].reshape(len(apn), 6000).astype(bool)

    # Artifact mask (atfs)
    atfs = np.zeros(int(t_end*100))
    atfs[idx_qrs[qrs == '|']] = 1
    atfs = atfs[3000: -3000].reshape(len(apn), 6000).astype(bool)

    return [apn, ecg, r_peaks, atfs]


def get_ecg(filename):
    # Method 2
    # signals, fields = wfdb.rdsamp(filename)
    record = wfdb.rdrecord(filename)
    signal = record.p_signal
    fs = record.fs
    return signal, fs


def get_apn_train(filename):
    # Read apn annotation for training data
    # apn: Numpy array with 0 indicates N (non-apnea) and 1 indicates A (apnea)
    # fs: Sampling frequency
    # t_apn: Time of each apn annotation
    annotation = wfdb.rdann(filename, extension='apn')
    apn = annotation.symbol
    fs = annotation.fs
    t_apn = annotation.sample / fs

    assert len(np.unique(np.diff(t_apn))) == 1, 'Un-uniform label intervals'
    assert t_apn[0] == 0, 'APN label does not start from zero'
    d_apn = {'N': 0, 'A': 1}
    apn = np.array([d_apn[str] for str in apn]).astype(bool)
    
    return apn, fs, t_apn


def get_qrs(filename):
    annotation = wfdb.rdann(filename, extension='qrs')
    qrs = annotation.symbol
    fs = annotation.fs
    t = annotation.sample / fs
    return qrs, fs, t


def get_additional_info(filename):
    try:
        record = wfdb.rdrecord(filename + 'r')
        signal = record.p_signal  # Resp C, Resp A, Resp N, SpO2
        fs = record.fs
        return signal, fs
    except:
        return None, None


