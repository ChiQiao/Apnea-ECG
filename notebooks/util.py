import matplotlib.pyplot as plt
import numpy as np


def ecg_diagnose(apn):
    """ Diagnose severity of sleep apnea

        Parameters
        ----------
        apn: list
            Apnea label for each minute

        Returns
        -------
        Severity group
    """
    # Total minutes of apnea
    apnea_total = sum(apn)

    # Maximum hourly Apnea Index
    total_hour = int(len(apn) / 60)
    y_pred_hourly = np.reshape(apn[: total_hour * 60], (total_hour, 60))
    AI_hourly = y_pred_hourly.sum(axis=1)
    # If data in the last hour exceed 30 minutes, then convert to hourly result
    y_pred_left = apn[total_hour * 60 :]
    # if len(y_pred_left) >= 30:
    #     AI_hourly = np.append(AI_hourly, sum(y_pred_left) * 60 / len(y_pred_left))
    #     total_hour += 1
    AI_max = AI_hourly.max()

    if AI_max >= 10 and apnea_total >= 100:
        return "A"
    elif AI_max >= 5 and apnea_total >= 5:
        return "B"
    else:
        return "C"


def get_normal_segment_idx(ecg, ratio_lb, ratio_ub, diagPlot=False):
    """ Detect abnormal ecg recording segments

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
    """
    ecg_sd = ecg.std(axis=1)
    ecg_sd_med = np.median(ecg_sd)
    idx_valid = (ecg_sd < ecg_sd_med * ratio_ub) & (ecg_sd > ecg_sd_med * ratio_lb)

    if diagPlot:
        plt.figure()
        for i in range(ecg.shape[0]):
            if idx_valid[i] == 0:
                plt.plot(np.arange(6000) + 6000 * i, ecg[i, :], "r-")
            else:
                plt.plot(np.arange(6000) + 6000 * i, ecg[i, :], "b-")
        # plt.show()

    return idx_valid
