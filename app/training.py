import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.models as models
import mlflow
import mlflow.pytorch

from evaluation import evaluate


def build_model(num_classes=5):
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    # model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
    # model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    return model


def train_model(train_ds, valid_ds, device="cuda", epochs=5, batch_size=16, lr=1e-4):

    train_loader = DataLoader(
    train_ds,
    batch_size=16,
    shuffle=True,
    num_workers=8,
    pin_memory=True
    )
    valid_loader = DataLoader(valid_ds, batch_size=16, shuffle=False, num_workers=8, pin_memory=True)

    with mlflow.start_run():

        mlflow.log_param("model", "resnet50")
        #mlflow.log_param("model", "densenet121")
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("learning_rate", lr)

        model = build_model()
        model = model.to(device)

        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)

        best_auc = 0.0

        for epoch in range(epochs):
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
                torch.save(model.state_dict(), "best_model.pt")

        mlflow.pytorch.log_model(model, "model")

    return model