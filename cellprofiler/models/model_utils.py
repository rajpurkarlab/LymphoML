import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pprint
import seaborn as sns
from scipy.misc import derivative
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

###############################################################################
# Utilities to Select Feature Groups
###############################################################################

def get_features_of_type(all_features, feature_type):
    return list(filter(lambda s : s.split("_")[0] == feature_type, all_features))

def get_area_shape_features(all_features, feature_type):
    assert(feature_type == "nucleiAreaShape" or feature_type == "cytoplasmAreaShape")
    all_area_shape_features = get_features_of_type(all_features, feature_type)
    # These are spatial features, so we don't include them for Section 1.1 experiments.
    to_remove = ["BoundingBoxMaximum", "BoundingBoxMinimum", "Center", "CentralMoment", "SpatialMoment"]
    area_shape_features = []
    for feature in all_area_shape_features:
        if feature.split("_")[1] not in to_remove:
            area_shape_features.append(feature)
    return area_shape_features

def get_location_features(all_features, feature_type):
    assert(feature_type == "nucleiLocation" or feature_type == "cytoplasmLocation")
    all_location_features = get_features_of_type(all_features, feature_type)
    # These are spatial features, so we don't include them for Section 1.1 experiments.
    to_remove = ["Center"]
    location_features = []
    for feature in all_location_features:
        if feature.split("_")[1] not in to_remove:
            location_features.append(feature)
    return location_features

def get_nuclei_morphological_features(all_features):
    area_shape_features = get_area_shape_features(all_features, "nucleiAreaShape")
    return area_shape_features

def get_nuclei_intensity_features(all_features):
    area_shape_features = get_area_shape_features(all_features, "nucleiAreaShape")
    location_features = get_location_features(all_features, "nucleiLocation")
    intensity_features = get_features_of_type(all_features, "nucleiIntensity")
    return area_shape_features + location_features + intensity_features

def get_nuclei_cytoplasm_morphological_features(all_features):
    n_area_shape_features = get_features_of_type(all_features, "nucleiAreaShape")
    n_children_features = get_features_of_type(all_features, "nucleiChildren")
    
    c_area_shape_features = get_features_of_type(all_features, "cytoplasmAreaShape")
    c_parent_features = get_features_of_type(all_features, "cytoplasmParent")
    
    return (n_area_shape_features + n_children_features + c_area_shape_features + c_parent_features)

###############################################################################
# Model Utilities
###############################################################################

pp = pprint.PrettyPrinter(indent=4)

def get_processed_df_splits(train_features_df, val_features_df, test_features_df, feature_cols):
    train_features_df = train_features_df[feature_cols + ["patient_id", "label", "count"]].dropna().reset_index(drop=True)
    val_features_df = val_features_df[feature_cols + ["patient_id", "label", "count"]].dropna().reset_index(drop=True)
    test_features_df = test_features_df[feature_cols + ["patient_id", "label", "count"]].dropna().reset_index(drop=True)
    
    return (train_features_df, val_features_df, test_features_df)

def get_splits(train_features_df, val_features_df, test_features_df,
    feature_cols, enable_dlbcl_classification=False, enable_normalization=True):
    (train_features_df, val_features_df, test_features_df) = get_processed_df_splits(
        train_features_df, val_features_df,test_features_df, feature_cols)
    train_df = pd.concat([train_features_df, val_features_df])
    test_df = test_features_df
    X_train = train_df[feature_cols].astype(np.float32)
    y_train = train_df["label"]
    X_test = test_df[feature_cols].astype(np.float32)
    y_test = test_df["label"]
    
    if enable_dlbcl_classification:
        y_train = y_train.apply(lambda l : 0 if l == 0 else 1)
        y_test = y_test.apply(lambda l : 0 if l == 0 else 1)
    
    if enable_normalization:
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
    return (X_train, y_train, X_test, y_test, scaler)

def compute_accuracy(preds, labels):
    return sum(preds == labels) / len(labels)

def get_confusion_matrix(y_true, y_pred):
    return confusion_matrix(y_true, y_pred)

def plot_confusion_matrix(cnf_matrix, num_classes):
    array = cnf_matrix.tolist()
    df_cm = pd.DataFrame(array, index = [i for i in range(num_classes)],
                      columns = [i for i in range(num_classes)])
    plt.figure(figsize = (8,8))
    sns.heatmap(df_cm, annot=True)

def get_f1_scores(y_true, y_pred):
    micro_f1 = f1_score(y_true, y_pred, average='macro')
    macro_f1 = f1_score(y_true, y_pred, average='micro')
    weighted_f1 = f1_score(y_true, y_pred, average='weighted')
    return (micro_f1, macro_f1, weighted_f1)

def build_metrics_dict(train, train_acc, test_acc, macro_f1, micro_f1, weighted_f1, cnf_matrix):
    metrics = dict()
    metrics["train_acc"] = train_acc
    if train:
        metrics["val_acc"] = test_acc
    else:
        metrics["test_acc"] = test_acc
    metrics["macro_f1"] = macro_f1
    metrics["micro_f1"] = micro_f1
    metrics["weighted_f1"] = weighted_f1
    metrics["cnf_matrix"] = cnf_matrix
    return metrics

def get_core_metrics(features_df, preds_patch, enable_dlbcl_classification=False):
    features_df["preds_patch"] = preds_patch
    y_core = features_df.groupby("patient_id")["label"].agg(pd.Series.mode)
    if enable_dlbcl_classification:
        y_core = y_core.apply(lambda l : 0 if l == 0 else 1)
    preds_core = features_df.groupby("patient_id")["preds_patch"].agg(lambda x: pd.Series.mode(x)[0])
    accuracy = compute_accuracy(preds_core, y_core)
    return (y_core, preds_core, accuracy)

def get_core_metrics_mean(features_df, pred_probs, num_classes):
    classes = list(range(num_classes))
    pred_probs_df = pd.DataFrame(pred_probs, columns = classes)
    pred_probs_df["patient_id"] = features_df["patient_id"]
    pred_prob_df_aggregated = pred_probs_df.groupby("patient_id").aggregate(pd.DataFrame.mean)
    y_core = features_df.groupby("patient_id")["label"].agg(pd.Series.mode)
    preds_core = pred_prob_df_aggregated.idxmax(axis=1)
    accuracy = compute_accuracy(preds_core, y_core)
    return (y_core, preds_core, accuracy)

###############################################################################
# Light GBM Models Utilities
###############################################################################

def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / (np.sum(exp_x, axis=1, keepdims=True) + 1e-6)

def get_lgb_pred_probs(model, X):
    return softmax(model.predict(X))

def get_lgb_preds(model, X):
    preds = np.argmax(get_lgb_pred_probs(model, X), axis=1)
    return preds

def eval_metric(num_classes):
    def accuracy_eval_metric(preds, dtrain):
        labels = dtrain.label
        preds = preds.reshape(num_classes, -1).T
        preds = preds.argmax(axis=1)
        accuracy = accuracy_score(labels, preds)
        return 'accuracy', accuracy, False
    return accuracy_eval_metric

def focal_loss_lgb(y_pred, dtrain, alpha, gamma, num_class):
    """
    Focal Loss for lightgbm

    Parameters:
    -----------
    y_pred: numpy.ndarray
        array with the predictions
    dtrain: lightgbm.Dataset
    alpha, gamma: float
        See original paper https://arxiv.org/pdf/1708.02002.pdf
    num_class: int
        number of classes
    """
    a,g = alpha, gamma
    y_true = dtrain.label
    # N observations x num_class arrays
    y_true = np.eye(num_class)[y_true.astype('int')]
    y_pred = y_pred.reshape(-1,num_class, order='F')
    # alpha and gamma multiplicative factors with BCEWithLogitsLoss
    def fl(x,t):
        p = 1/(1+np.exp(-x))
        return -( a*t + (1-a)*(1-t) ) * (( 1 - ( t*p + (1-t)*(1-p)) )**g) * ( t*np.log(p)+(1-t)*np.log(1-p) )
    partial_fl = lambda x: fl(x, y_true)
    grad = derivative(partial_fl, y_pred, n=1, dx=1e-6)
    hess = derivative(partial_fl, y_pred, n=2, dx=1e-6)
    # flatten in column-major (Fortran-style) order
    return grad.flatten('F'), hess.flatten('F')

def focal_loss_lgb_eval_error(y_pred, dtrain, alpha, gamma, num_class):
    """
    Focal Loss for lightgbm

    Parameters:
    -----------
    y_pred: numpy.ndarray
        array with the predictions
    dtrain: lightgbm.Dataset
    alpha, gamma: float
        See original paper https://arxiv.org/pdf/1708.02002.pdf
    num_class: int
        number of classes
    """
    a,g = alpha, gamma
    y_true = dtrain.label
    y_true = np.eye(num_class)[y_true.astype('int')]
    y_pred = y_pred.reshape(-1, num_class, order='F')
    p = 1/(1+np.exp(-y_pred))
    loss = -( a*y_true + (1-a)*(1-y_true) ) * (( 1 - ( y_true*p + (1-y_true)*(1-p)) )**g) * ( y_true*np.log(p)+(1-y_true)*np.log(1-p) )
    # a variant can be np.sum(loss)/num_class
    return 'focal_loss', np.mean(loss), False

