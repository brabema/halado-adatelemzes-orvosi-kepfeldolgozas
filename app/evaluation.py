import torch
from sklearn.metrics import roc_auc_score
import mlflow


def evaluate(model, dataloader, device="cuda", log_prefix=None):
    model.eval()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)

            outputs = model(images)
            probs = torch.sigmoid(outputs)
            all_preds.append(probs.cpu())
            all_targets.append(labels)

    y_pred = torch.cat(all_preds).numpy()
    y_true = torch.cat(all_targets).numpy()

    try:
        auc = roc_auc_score(y_true, y_pred, average="macro")
    except ValueError:
        auc = 0.0

    if log_prefix:
        mlflow.log_metric(f"{log_prefix}_auc", auc)

    return auc