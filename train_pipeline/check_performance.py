import torch
import os
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from torch.utils.data import DataLoader
import torch.nn as nn
from torchvision import datasets


def save_model_state(state_dict_epoch: dict):
    if not os.path.exists("./model_state"):
        os.mkdir("./model_state")

    current_date = datetime.now().strftime("%Y-%m-%d_%H-%M")
    filename = f"./model_state/model_{current_date}.pt"

    torch.save(state_dict_epoch["state_dict"], filename)


def load_model_state():
    current_dir = os.getcwd()
    model_state_dir = os.path.join(current_dir, "model_state")
    if os.path.exists(model_state_dir):
        files = os.listdir(model_state_dir)
        if len(files) > 0:
            file_paths = [os.path.join(model_state_dir, file) for file in files]
            latest_file = max(file_paths, key=os.path.getctime)
            model_state_path = os.path.join(model_state_dir, latest_file)
            print(f"Loading model state from {model_state_path}")
            return torch.load(model_state_path)
    return None


def check_performance(model: nn.Module, test_dataset: datasets):
    model.to(torch.device("cpu"))
    model.eval()
    batch_size = 1
    validation_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    true_labels = []
    predicted_labels = []
    with torch.no_grad():
        for images, labels in validation_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            true_labels.append(labels.item())
            predicted_labels.append(predicted.item())
    accuracy = accuracy_score(true_labels, predicted_labels)
    balanced_accuracy = balanced_accuracy_score(true_labels, predicted_labels)
    cm = confusion_matrix(true_labels, predicted_labels)
    print(f"Accuracy of the network on the 10000 test images: {100 * accuracy:.2f} %")
    print(
        f"Balanced Accuracy of the network on the 10000 test images: {100 * balanced_accuracy:.2f} %"
    )
    ConfusionMatrixDisplay(
        cm,
        display_labels=[
            "Airplane",
            "Automobile",
            "Bird",
            "Cat",
            "Deer",
            "Dog",
            "Frog",
            "Horse",
            "Ship",
            "Truck",
        ],
    ).plot(values_format="d")
    plt.savefig("confusion_matrix.png")
