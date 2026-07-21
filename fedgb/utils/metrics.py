import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from munkres import Munkres


def normalize_required_metrics(metrics, task=None, num_classes=None):
    """Keep user metrics, while enforcing project-level reporting metrics."""
    normalized = []
    for metric in metrics or []:
        if metric not in normalized:
            normalized.append(metric)

    required = []
    if task in ["graph_reg"]:
        required = ["mae", "rmse"]
    elif task == "link_pred":
        required = ["auc", "ap"]
    elif task in ["graph_cls", "node_cls"]:
        if num_classes == 2:
            required = ["auc", "ap"]
        elif num_classes is not None and num_classes > 2:
            required = ["accuracy", "f1"]

    for metric in required:
        if metric not in normalized:
            normalized.append(metric)
    return normalized


def compute_supervised_metrics(metrics, logits, labels, suffix):
    """
    Compute various supervised learning metrics based on provided logits and labels.

    Args:
        metrics (list of str): List of metric names to compute.
        logits (torch.Tensor): Predicted logits from the model.
        labels (torch.Tensor): Ground truth labels.
        suffix (str): Suffix to append to the metric names in the result dictionary.

    Raises:
        ValueError: If AUC is requested for multi-class classification.

    Returns:
        dict: Dictionary containing the computed metrics.
    """
    result = {}
    
    if logits.dtype == torch.long: # clustering labels
        probs = logits.cpu().numpy()
        preds = logits.cpu().numpy()
        np_labels = labels.cpu().numpy()
    else: # classification logits
        if logits.dim() == 1: # binary
            probs = F.sigmoid(logits)
            preds = (probs > 0.5).long()
        else:
            probs = F.softmax(logits, dim=1) # multi
            _, preds = torch.max(logits, 1)
            
        probs = probs.cpu().numpy()
        preds = preds.cpu().numpy()
        np_labels = labels.cpu().numpy()
        
    if "accuracy" in metrics:
        result[f"accuracy_{suffix}"] = accuracy_score(np_labels, preds)
    
    if "precision" in metrics:
        result[f"precision_{suffix}"] = precision_score(np_labels, preds, average='weighted', zero_division=0)

    if "recall" in metrics:
        result[f"recall_{suffix}"] = recall_score(np_labels, preds, average='weighted', zero_division=0)
        
    if "f1" in metrics:
        result[f"f1_{suffix}"] = f1_score(np_labels, preds, average='weighted', zero_division=0)
        
    if "auc" in metrics:
        if np_labels.max() > 1:
            raise ValueError("AUC is not directly supported for multi-class classification.")
        positive_scores = probs[:, 1] if probs.ndim > 1 and probs.shape[1] > 1 else probs
        result[f"auc_{suffix}"] = 0.5 if len(np.unique(np_labels)) < 2 else roc_auc_score(np_labels, positive_scores)

    if "ap" in metrics:
        positive_scores = probs[:, 1] if probs.ndim > 1 and probs.shape[1] > 1 else probs
        result[f"ap_{suffix}"] = average_precision_score(np_labels, positive_scores)
        
    # Clustering specific metrics
    if "clustering_accuracy" in metrics:
        result["clustering_accuracy"] = clustering_acc(np_labels, preds)
    
    if "nmi" in metrics:
        result["nmi"] = normalized_mutual_info_score(np_labels, preds)

    if "ari" in metrics:
        result["ari"] = adjusted_rand_score(np_labels, preds)
        
    return result


def compute_regression_metrics(metrics, preds, labels, suffix):
    """
    Compute graph regression metrics. Lower values are better for all metrics here.
    """
    result = {}
    preds = preds.float().view(labels.shape)
    labels = labels.float()
    diff = preds - labels

    if "mae" in metrics:
        result[f"mae_{suffix}"] = torch.mean(torch.abs(diff)).item()
    if "mse" in metrics:
        result[f"mse_{suffix}"] = torch.mean(diff ** 2).item()
    if "rmse" in metrics:
        result[f"rmse_{suffix}"] = torch.sqrt(torch.mean(diff ** 2)).item()

    return result





# similar to https://github.com/karenlatong/AGC-master/blob/master/metrics.py
def clustering_acc(y_true, y_pred):
    y_true = y_true - np.min(y_true)

    l1 = list(set(y_true))
    numclass1 = len(l1)

    l2 = list(set(y_pred))
    numclass2 = len(l2)

    ind = 0
    if numclass1 != numclass2:
        for i in l1:
            if i in l2:
                pass
            else:
                y_pred[ind] = i
                ind += 1

    l2 = list(set(y_pred))
    numclass2 = len(l2)

    if numclass1 != numclass2:
        print("error")
        return

    cost = np.zeros((numclass1, numclass2), dtype=int)
    for i, c1 in enumerate(l1):
        mps = [i1 for i1, e1 in enumerate(y_true) if e1 == c1]
        for j, c2 in enumerate(l2):
            mps_d = [i1 for i1 in mps if y_pred[i1] == c2]
            cost[i][j] = len(mps_d)

    # match two clustering results by Munkres algorithm
    m = Munkres()
    cost = cost.__neg__().tolist()
    indexes = m.compute(cost)

    # get the match results
    new_predict = np.zeros(len(y_pred))
    for i, c in enumerate(l1):
        # correponding label in l2:
        c2 = l2[indexes[i][1]]

        # ai is the index with label==c2 in the pred_label list
        ai = [ind for ind, elm in enumerate(y_pred) if elm == c2]
        new_predict[ai] = c

    acc = accuracy_score(y_true, new_predict)

    return acc
