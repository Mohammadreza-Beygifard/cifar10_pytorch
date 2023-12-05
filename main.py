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


class StoreTrueAction(argparse.Action):
    def __init__(self, option_strings, dest, default=False, required=False, help=None):
        super(StoreTrueAction, self).__init__(
            option_strings=option_strings,
            dest=dest,
            nargs=0,
            const=True,
            default=default,
            required=required,
            help=help,
        )

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, self.const)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--train", action=StoreTrueAction, default=False)
    parser.add_argument(
        "-c", "--check-performance", action=StoreTrueAction, default=False
    )
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
