import torch
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

@torch.no_grad()
def evaluate_model(data_loader,
                model_A, 
                model_B=None,
                threshold=0.5,
                device='cuda'):
    
    model_A.eval()
    if model_B is not None: model_B.eval()
    total_f1_micro, total_f1_macro, total_precision_micro, total_recall_micro, total_precision_macro, total_recall_macro = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    total_subset_acc, total_sample_acc = 0.0, 0.0
    n_batches = 0

    for img, label in data_loader:
        img, label = img.to(device), label.to(device)

        probs_A, _ = model_A(img)

        if model_B is None:
            probs = probs_A
        else:
            probs_B, _ = model_B(img)
            probs = (probs_A + probs_B) / 2.0
            
        preds = (probs > threshold).float()
        f1_micro, f1_macro, precision_micro, recall_micro, precision_macro, recall_macro, subset_acc, sample_acc = multilabel_metrics_from_preds(preds, label)

        total_f1_micro += f1_micro
        total_f1_macro += f1_macro
        total_precision_micro += precision_micro
        total_recall_micro += recall_micro
        total_precision_macro += precision_macro
        total_recall_macro += recall_macro
        total_subset_acc += subset_acc
        total_sample_acc += sample_acc
        n_batches += 1

    metrics = {
        "f1_micro": total_f1_micro / n_batches,
        "f1_macro": total_f1_macro / n_batches,
        "subset_acc": total_subset_acc / n_batches,
        "sample_acc": total_sample_acc / n_batches,
    }
    return metrics

def multilabel_metrics_from_preds(preds, labels):
    """
    Compute multilabel classification metrics from predicted and true labels.

    Args:
        preds (torch.Tensor): The predicted logits of shape (num_samples, num_classes).
        labels (torch.Tensor): The true binary labels of shape (num_samples, num_classes).
        threshold (float): The threshold to convert probabilities to binary predictions.

    Returns:
        dict: A dictionary containing the computed metrics (F1 score, accuracy, etc.).
    """


    f1_micro = f1_score(labels.detach().cpu(), preds.detach().cpu(), average='micro', zero_division=0)
    f1_macro = f1_score(labels.detach().cpu(), preds.detach().cpu(), average='macro', zero_division=0)

    precision_micro = precision_score(labels.detach().cpu(), preds.detach().cpu(), average='micro', zero_division=0)
    precision_macro = precision_score(labels.detach().cpu(), preds.detach().cpu(), average='macro', zero_division=0)

    recall_micro = recall_score(labels.detach().cpu(), preds.detach().cpu(), average='micro', zero_division=0)
    recall_macro = recall_score(labels.detach().cpu(), preds.detach().cpu(), average='macro', zero_division=0)

    subset_acc = accuracy_score(labels.detach().cpu(), preds.detach().cpu())
    sample_acc = (preds.detach().cpu() == labels.detach().cpu()).float().mean().item()

    return f1_micro, f1_macro, precision_micro, recall_micro, precision_macro, recall_macro, subset_acc, sample_acc