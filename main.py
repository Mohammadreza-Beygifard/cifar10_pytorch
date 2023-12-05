from data_pipeline.data_pipeline import get_cifar10, get_train_data
from inference_pipeline.inference import run_inference
from train_pipeline.train_pipeline import run_train
from train_pipeline.check_performance import (
    check_performance,
    save_model_state,
    load_model_state,
)
from models.vgg import vgg_a

import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--train", type=bool, default=False)
    parser.add_argument("-c", "--check-performance", type=bool, default=False)
    parser.add_argument("-p", "--path-image", type=str, default="")
    return parser.parse_args()


if __name__ == "__main__":
    """
    If you want to train the model, set --train=True
    """
    args = parse_args()
    model = vgg_a()

    if args.train or args.check_performance:
        original_train_dataset, test_dataset = get_cifar10()

    if args.train:
        print("Training model")
        train_dataset = get_train_data()
        model_state = run_train(model, train_dataset)
        save_model_state(model_state)

    early_drop_model_state = load_model_state()
    model.load_state_dict(early_drop_model_state)

    if args.check_performance:
        print("Checking performance")
        check_performance(model, test_dataset)

    if args.path_image:
        print("Running inference")
        print(f"Predicted class: {run_inference(model, args.path_image)}")
