import torch
import os
from datetime import datetime


def save_model_state(state_dict_epoch: dict):
    if not os.path.exists("./model_state"):
        os.mkdir("./model_state")

    current_date = datetime.now().strftime("%Y-%m-%d_%H-%M")
    filename = f"./model_state/model_{current_date}.pt"

    torch.save(state_dict_epoch["state_dict"], filename)


def load_model_state():
    if os.path.exists("./model_state"):
        files = os.listdir("./model_state")
        if len(files) > 0:
            latest_file = max(files, key=os.path.getctime)
            return torch.load(f"./model_state/{latest_file}")
    return None


def check_performance(model, test_dataset):
    model.to(torch.device("cpu"))
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_dataset:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(
        f"Accuracy of the network on the 10000 test images: {100 * correct / total} %"
    )
