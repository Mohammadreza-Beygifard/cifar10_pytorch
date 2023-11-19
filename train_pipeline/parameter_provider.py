import json
from dataclasses import dataclass


@dataclass
class OptimizerParameters:
    learning_rate: float
    weight_decay: float


@dataclass
class SchedulerParameters:
    active: bool
    gamma: float


@dataclass
class TrainingParameters:
    optimizer_parameters: OptimizerParameters
    scheduler_parameters: SchedulerParameters
    batch_size: int


def get_parameters():
    with open("training_parameters.json", "r") as f:
        parameters = json.load(f)
    return parameters


def get_optimizer_parameters():
    parameters = get_parameters()
    optimizer_parameters = OptimizerParameters(
        parameters["optimizer"]["learning_rate"],
        parameters["optimizer"]["weight_decay"],
    )

    return optimizer_parameters


def get_scheduler_parameters():
    parameters = get_parameters()
    scheduler_parameters = SchedulerParameters(
        parameters["scheduler"]["active"],
        parameters["scheduler"]["gamma"],
    )

    return scheduler_parameters


def get_training_parameters():
    parameters = get_parameters()
    training_parameters = TrainingParameters(
        get_optimizer_parameters(),
        get_scheduler_parameters(),
        parameters["batch_size"],
    )

    return training_parameters
