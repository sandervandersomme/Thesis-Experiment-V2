import numpy as np
import torch
from tslearn.metrics import dtw as dtw_distance
from sklearn.metrics import precision_score, recall_score, mean_squared_error, roc_auc_score

from src.utils import set_device

def reidentification_risk(syndata: torch.Tensor, train_data: torch.Tensor, dtw_threshold: float):
    device = set_device()
    
    syndata = syndata[:, None, :, :]
    t_data = train_data[None, :, :, :]
    
    # Calculate DTW distances
    distances = torch.zeros(syndata.size(0), train_data.size(0), device=device)
    for i in range(syndata.size(0)):
        for j in range(train_data.size(0)):
            distances[i, j] = dtw_distance(syndata[i].cpu().numpy().squeeze(),
                                           t_data[0, j].cpu().numpy().squeeze())
    
    # Determine membership based on DTW distance threshold
    inferred_membership = (distances.min(dim=1).values < dtw_threshold).int()

    # Convert tensors to numpy arrays
    inferred_membership_np = inferred_membership.cpu().numpy()
    
    # Assuming labels for synthetic data (syndata) based on your context
    labels_np = np.ones(syndata.size(0))  # Assuming all are potential reidentification risks

    # Calculate precision and recall
    precision = precision_score(labels_np, inferred_membership_np)
    recall = recall_score(labels_np, inferred_membership_np)

    # Calculate mean squared error
    mse = mean_squared_error(labels_np, inferred_membership_np)

    # Calculate AUC-ROC if both classes are present
    if len(np.unique(labels_np)) == 1:
        auc_roc = None  # or any default value you choose
    else:
        auc_roc = roc_auc_score(labels_np, inferred_membership_np)

    return {
        "precision": precision, 
        "recall": recall, 
        "mse": mse, 
        "auc_roc": auc_roc 
    }
