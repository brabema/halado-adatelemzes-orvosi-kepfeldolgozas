import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.models as models
import mlflow
import mlflow.pytorch
import os
from evaluation import evaluate

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def build_model(model_name="resnet50", num_classes=5, dropout=0.0):
    if model_name == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes)
        )

    elif model_name == "densenet121":
        model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        in_features = model.classifier.in_features
        model.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes)
        )

    return model
    
def train_model(train_ds, valid_ds, device="cuda", config):
    set_seed(config["seed"])
    
    with mlflow.start_run():
        mlflow.log_params(config)

        model = build_model(
            model_name=config["model"],
            dropout=config["dropout"]
        ).to(device)

        subset_size = int(len(train_ds) * config["data_fraction"])
        train_ds = torch.utils.data.Subset(train_ds, range(subset_size))

        train_loader = DataLoader(
            train_ds,
            batch_size=config["batch_size"],
            shuffle=True,
            num_workers=8,
            pin_memory=True
        )

        valid_loader = DataLoader(
            valid_ds, 
            batch_size=config["batch_size"], 
            shuffle=False, 
            num_workers=8, 
            pin_memory=True
        )


        model = build_model(modelName)
        model = model.to(device)

        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=config["lr"])

        best_auc = 0.0

        for epoch in range(config["epochs"]):
            model.train()
            running_loss = 0.0

            for images, labels in train_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            avg_loss = running_loss / len(train_loader)

            val_auc = evaluate(model, valid_loader, device)

            print(f"[Epoch {epoch+1}] Loss: {avg_loss:.4f} | Val AUC: {val_auc:.4f}")

            mlflow.log_metric("train_loss", avg_loss, step=epoch)
            mlflow.log_metric("val_auc", val_auc, step=epoch)

            if val_auc > best_auc:
                best_auc = val_auc
                print(f"Saving the best model, the acc: {best_auc}")
                save_path = "./checkpoints"
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save(model.state_dict(), "best_model.pt")

        mlflow.pytorch.log_model(model, "model")

    return model