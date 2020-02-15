import numpy as np
import wfdb
from pathlib import Path


def single_file_res(data_folder, file, apn, t_end):
    """ Extract information from a set of wfdb files assuming 100 Hz sampling frequency

    Parameters
    ----------
    data_folder: str
        Folder path that contains the file
    file: str
        File name without extension, as several extensions will be extracted
    apn: list
        Apnea label for each minute, from get_apn_train for training data 
        or event-2-answers.pkl for testing data
    t_end: scalar
        Time of the last apn label in seconds

    Returns
    -------
    apn: list
        Truncated apn labels, paired with ecg, r_peaks, and atfs
    ecg: numpy 2D array
        Each row represent one-minute ECG recording
    r_peaks: numpy 2D boolean array
        Mask indicating R peaks for the ecg, same size as ecg
    atfs: numpy 2D boolean array
        Mask indicating QRS-artefacts for the ecg, same size as ecg
    """

    def get_ecg(filename):
        # Method 2
        # signals, fields = wfdb.rdsamp(filename)
        record = wfdb.rdrecord(filename)
        signal = record.p_signal
        fs = record.fs
        return signal, fs

    def get_qrs(filename):
        annotation = wfdb.rdann(filename, extension="qrs")
        qrs = annotation.symbol
        fs = annotation.fs
        t = annotation.sample / fs
        return qrs, fs, t

    apn = apn[1:-1]  # Remove first and last APN label

    # ECG data (ecg)
    ecg, _ = get_ecg(str(Path(data_folder) / file))
    ecg = np.array(ecg).flatten()
    assert len(ecg) >= t_end * 100, "Period of ecg shorter than label"
    ecg = ecg[: int(t_end * 100)]  # Discard the tail
    # Discard first 30 sec and last 30 sec, so apn is at middle
    ecg = ecg[3000:-3000].reshape(len(apn), 6000)

    # QRS label
    qrs, _, t_qrs = get_qrs(str(Path(data_folder) / file))
    qrs = np.array(qrs)
    qrs = qrs[t_qrs <= t_end]
    t_qrs = t_qrs[t_qrs <= t_end]
    idx_qrs = (t_qrs * 100).astype(int)

    # R peak mask (r_peaks)
    r_peaks = np.zeros(int(t_end * 100))
    r_peaks[idx_qrs[qrs == "N"]] = 1
    r_peaks = r_peaks[3000:-3000].reshape(len(apn), 6000).astype(bool)

    # Artifact mask (atfs)
    atfs = np.zeros(int(t_end * 100))
    atfs[idx_qrs[qrs == "|"]] = 1
    atfs = atfs[3000:-3000].reshape(len(apn), 6000).astype(bool)

    return apn, ecg, r_peaks, atfs


def get_apn_train(filename):
    """ Read apn annotation for training data

    Parameters
    ----------
    filename: str
        Path of the .apn file
        
    Returns
    -------
    apn: numpy array 
        0 indicates N (non-apnea) and 1 indicates A (apnea)
    t_apn: numpy array
        Time of each apn annotation
    """
    annotation = wfdb.rdann(filename, extension="apn")
    apn = annotation.symbol
    fs = annotation.fs
    t_apn = annotation.sample / fs

    assert len(np.unique(np.diff(t_apn))) == 1, "Un-uniform label intervals"
    assert t_apn[0] == 0, "APN label does not start from zero"
    d_apn = {"N": 0, "A": 1}
    apn = np.array([d_apn[str] for str in apn]).astype(bool)

    return apn, t_apn
