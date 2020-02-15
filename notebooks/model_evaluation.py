import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import interp
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics

import util


def model_evaluation_CV(
    mdl, df, file_df, feature_col, n=4, normalize=True, plot_roc=False
):
    """ Evaluate model accuracy using Stratified K-fold CV
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
    """
    # Initialize
    auc_val = []
    group_res = file_df.copy(deep=True).set_index("file")
    group_res.rename(columns={"group": "true"}, inplace=True)
    minute_res = {}
    if plot_roc:
        tprs, aucs = [], []
        mean_fpr = np.linspace(0, 1, 100)
        _, ax = plt.subplots()

    skf = StratifiedKFold(n_splits=n)
    for idx_train, idx_val in skf.split(file_df, file_df["group"]):
        file_train, file_val = (
            file_df.loc[idx_train, "file"],
            file_df.loc[idx_val, "file"],
        )
        X_train, y_train = (
            df.loc[df.file.isin(file_train), feature_col],
            df.loc[df.file.isin(file_train), "apn"],
        )
        X_val, y_val = (
            df.loc[df.file.isin(file_val), feature_col],
            df.loc[df.file.isin(file_val), "apn"],
        )

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
            X_val, y_val = (
                df.loc[df.file.isin([file]), feature_col],
                df.loc[df.file.isin([file]), "apn"],
            )
            X_val = scaler.transform(X_val) if normalize else X_val
            # Group prediction
            y_pred = mdl.predict(X_val)
            group_res.loc[file, "pred"] = util.ecg_diagnose(y_pred)
            group_res.loc[file, "true"] = util.ecg_diagnose(
                y_val.values
            )  # Original group might be wrong (a10 is identified as B)
            # Minute-wise prediction probability
            y_pred_prob = mdl.predict_proba(X_val)
            minute_res[file] = np.vstack((y_val, y_pred_prob[:, 1]))

    minute_auc = np.mean(auc_val)
    group_auc_macro, group_f1_best, thres_best, multiclass_auc = eval_multiclass_auc(
        group_res, minute_res
    )

    res = {
        "minute_auc_mean": minute_auc,
        "minute_auc_cv": auc_val,
        "group_auc": group_auc_macro,
        "group_f1_best": group_f1_best,
        "thres_best": thres_best,
        "minute_detail": minute_res,
        "group_detail": group_res,
        "multiclass_auc": multiclass_auc,
    }

    if plot_roc:
        ax.plot(
            [0, 1], [0, 1], linestyle="--", lw=2, color="r", label="Chance", alpha=0.8
        )

        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = metrics.auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)
        ax.plot(
            mean_fpr,
            mean_tpr,
            color="b",
            label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
            lw=2,
            alpha=0.8,
        )

        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        ax.fill_between(
            mean_fpr,
            tprs_lower,
            tprs_upper,
            color="grey",
            alpha=0.2,
            label=r"$\pm$ 1 std. dev.",
        )

        ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05])
        ax.legend(loc="lower right")
        plt.show()

        res["mean_fpr_minute"] = mean_fpr
        res["mean_tpr_minute"] = mean_tpr
        res["mean_auc_minute"] = mean_auc

    return res


def model_evaluation_test(mdl, df, file_df, feature_col, scaler, thres):
    """ Evaluate model accuracy on testing data
        
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
    """
    # Evaluate model accuracy using Stratified K-fold CV
    # Note: Stratification is based on patient group (A, B, C), and then samples are formed

    # Initialize
    group_res = file_df.copy(deep=True).set_index("file")
    group_res.rename(columns={"group": "true"}, inplace=True)
    minute_res = {}

    # Normalize df
    df_norm = pd.DataFrame(scaler.transform(df[feature_col]), columns=feature_col)
    df = df_norm.join(df[["apn", "file", "group"]])

    # Overall AUC
    minute_auc = metrics.roc_auc_score(
        df["apn"], mdl.predict_proba(df[feature_col])[:, 1]
    )

    # Accuracy for group diagnosis
    for file in file_df["file"]:
        X, y = (
            df.loc[df.file.isin([file]), feature_col],
            df.loc[df.file.isin([file]), "apn"],
        )
        # Group prediction
        y_pred = (mdl.predict_proba(X)[:, 1] > thres).astype(int)
        group_res.loc[file, "pred"] = util.ecg_diagnose(y_pred)
        group_res.loc[file, "true"] = util.ecg_diagnose(
            y.values
        )  # Original group might be wrong (a10 is identified as B)
        minute_res[file] = np.vstack((y, y_pred))

    res_detail = {
        "group_res": group_res,
        "minute_auc": minute_auc,
        "minute_detail": minute_res,
        "group_detail": group_res,
    }

    return res_detail


def eval_multiclass_auc(group_res, minute_res):
    """ Calculate multiclass ROC

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
    """
    multiclass_auc = {
        "fpr_A": [],
        "tpr_A": [],
        "fpr_B": [],
        "tpr_B": [],
        "fpr_C": [],
        "tpr_C": [],
        "f1_macro": [],
    }
    group_true = group_res["true"].values

    # AUC
    thres_all = np.linspace(0, 1, 101)
    for thres in thres_all:
        group_pred = np.array(
            [
                util.ecg_diagnose(minute_res[patient][1, :] > thres)
                for patient in group_res.index
            ]
        )
        multiclass_auc["f1_macro"].append(
            metrics.f1_score(group_true, group_pred, average="macro")
        )
        for group in list("ABC"):
            temp_true = group_true == group
            temp_pred = group_pred == group

            # calculate tpr & fpr
            tn, fp, fn, tp = metrics.confusion_matrix(temp_true, temp_pred).ravel()
            fpr = fp / (tn + fp)
            tpr = tp / (tp + fn)

            multiclass_auc["fpr_" + group].append(fpr)
            multiclass_auc["tpr_" + group].append(tpr)

    # Macro avg. of AUC for class A & C (B does not form intact ROC curve)
    f1_macro_opt = np.max(multiclass_auc["f1_macro"])
    thres_opt = thres_all[np.argmax(multiclass_auc["f1_macro"])]
    auc_macro = np.mean(
        [
            metrics.auc(multiclass_auc["fpr_A"], multiclass_auc["tpr_A"]),
            metrics.auc(multiclass_auc["fpr_C"], multiclass_auc["tpr_C"]),
        ]
    )

    return auc_macro, f1_macro_opt, thres_opt, multiclass_auc
