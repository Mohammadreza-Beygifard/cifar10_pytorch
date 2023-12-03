from data_pipeline.data_pipeline import get_cifar10, get_train_data
from train_pipeline.train_pipeline import run_train
from train_pipeline.check_performance import (
    check_performance,
    save_model_state,
    load_model_state,
)
from models.vgg import vgg_a


if __name__ == "__main__":
    original_train_dataset, test_dataset = get_cifar10()
    # train_dataset = get_train_data()
    model = vgg_a()
    # model_state = run_train(model, train_dataset)
    # check_performance(model, test_dataset)
    # save_model_state(model_state)
    early_drop_model_state = load_model_state()
    model.load_state_dict(early_drop_model_state)
    print("Check performance, early dropped model")
    check_performance(model, test_dataset)
