from data_pipeline.data_pipeline import get_cifar10, get_train_data
from train_pipeline.train_pipeline import plot_loss_accuracy, run_train
from models.vgg import vgg_a


if __name__ == "__main__":
    # create an example to test the plot function
    # plot_loss_accuracy(loss, accuracy, epochs)
    original_train_dataset, test_dataset = get_cifar10()
    train_dataset = get_train_data()
    model = vgg_a()
    run_train(model, train_dataset, 2)
    # print(model.eval())
    # train_model();
