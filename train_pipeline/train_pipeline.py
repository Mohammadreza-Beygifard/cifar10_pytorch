import matplotlib.pyplot as plt
import time
from torch.optim.lr_scheduler import ExponentialLR
import torch.nn as nn
import torchvision.datasets as datasets
import torch.optim as optim
from torch.utils.data import DataLoader
import torch

from train_pipeline.parameter_provider import (
    TrainingParameters,
    get_training_parameters,
)


def plot_loss_accuracy(loss: list, accuracy: list, epochs: list, silent: bool = False):
    plt.figure()
    plt.title("Cross Entropy Loss")
    plt.plot(epochs, loss, color="blue", label="loss")
    plt.plot(epochs, accuracy, color="green", label="accuracy")
    plt.legend(loc="upper left")
    if not silent:
        plt.savefig("loss_accuracy.png")


def update_state_epoch(epoch, loss_value, model_state, state_dict_epoch):
    """This method is used to capture the state when the loss has its minimum value"""
    if epoch == 0:
        state_dict_epoch["epoch"] = epoch
        state_dict_epoch["loss"] = loss_value
        state_dict_epoch["state_dict"] = model_state
    if epoch >= 1 and loss_value < state_dict_epoch["loss"]:
        state_dict_epoch["epoch"] = epoch
        state_dict_epoch["loss"] = loss_value
        state_dict_epoch["state_dict"] = model_state


def choose_device() -> torch.device:
    """Move the device to GPU if it is supported by the OS"""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def train(
    model: nn.Module,
    train_dataset: datasets,
    training_parameters: TrainingParameters,
    state_dict_epoch: dict,
    update_state_epoch: callable,
):
    device = choose_device()

    model.to(device)  # Use GPU if it is available, check the first cell for more info

    print(f"Starting training on {device}")

    epochs = range(training_parameters.num_epochs)

    # This is used for drawing the loss against the epochs
    loss_list = []

    # This is used for calculating the accuracy throw training
    correct = 0
    total = 0
    accuracy_list = []

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=training_parameters.optimizer_parameters.learning_rate,
        weight_decay=training_parameters.optimizer_parameters.weight_decay,
    )
    scheduler = ExponentialLR(
        optimizer, gamma=training_parameters.scheduler_parameters.gamma
    )

    batch_size = training_parameters.batch_size
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    start_time = time.time()

    # Define the beta value for the EMA (usually a value close to 1, e.g., 0.9)
    beta = 0.9

    # Initialize the EMA loss
    ema_loss = None

    for epoch in epochs:
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

        accuracy = correct / total
        accuracy_list.append(accuracy)
        # Calculate EMA loss
        if ema_loss is None:
            ema_loss = loss.item()
        else:
            ema_loss = beta * ema_loss + (1 - beta) * loss.item()
        loss_list.append(ema_loss)

        update_state_epoch(epoch, loss.item(), model.state_dict(), state_dict_epoch)

        if training_parameters.scheduler_parameters.active:
            scheduler.step()
        if (epoch + 1) % (training_parameters.num_epochs / 5) == 0 or epoch == 0:
            print(
                f"Epoch [{epoch+1}/{training_parameters.num_epochs}] - Loss: {loss.item():.4f} - Accuracy: {accuracy * 100:.2f}%"
            )

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.2f} seconds")

    print(f"""Minimum loss has happened at epoch number {state_dict_epoch["epoch"]}""")

    plot_loss_accuracy(loss_list, accuracy_list, epochs)


def custom_weight_init(module):
    """You can initialize your model weights by a custom method"""
    if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
        # Initialize weights using your custom logic
        nn.init.xavier_normal_(module.weight)


def run_train(model: nn.Module, train_dataset: datasets) -> dict:
    model.apply(custom_weight_init)

    state_dict_epoch = {"epoch": 0, "loss": 0, "state_dict": {}}

    training_parameters = get_training_parameters()

    train(
        model,
        train_dataset,
        training_parameters,
        state_dict_epoch,
        update_state_epoch,
    )
    return state_dict_epoch
