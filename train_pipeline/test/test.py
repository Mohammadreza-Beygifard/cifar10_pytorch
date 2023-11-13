import unittest
from unittest.mock import patch, MagicMock
import matplotlib.pyplot as plt
import torch
from train_pipeline.train_pipeline import (
    plot_loss_accuracy,
    update_state_epoch,
    choose_device,
    train,
)
from models.vgg import vgg_a


class TestPlotting(unittest.TestCase):
    def test_plot_loss_accuracy(self):
        # Create some dummy data
        loss = [0.5, 0.4, 0.3, 0.2, 0.1]
        accuracy = [0.7, 0.8, 0.85, 0.9, 0.95]
        epochs = [1, 2, 3, 4, 5]

        # Call the plotting function
        plot_loss_accuracy(loss, accuracy, epochs, silent=True)

        # No assertion here as we are just checking if the function runs without errors

    def test_update_state_epoch(self):
        # Initialize some dummy data
        epoch = 1
        loss_value = 0.1
        model_state = {"param1": 1, "param2": 2}
        state_dict_epoch = {"epoch": 0, "loss": float("inf"), "state_dict": {}}

        # Call the function with the dummy data
        update_state_epoch(epoch, loss_value, model_state, state_dict_epoch)

        # Assertions
        self.assertEqual(state_dict_epoch["epoch"], epoch)
        self.assertEqual(state_dict_epoch["loss"], loss_value)
        self.assertEqual(state_dict_epoch["state_dict"], model_state)

    def test_update_state_epoch_better_loss(self):
        # Initialize some dummy data with a better loss value
        epoch = 2
        loss_value = 0.05
        model_state = {"param1": 1, "param2": 2}
        state_dict_epoch = {
            "epoch": 1,
            "loss": 0.1,
            "state_dict": {"param1": 0, "param2": 0},
        }

        # Call the function with the dummy data
        update_state_epoch(epoch, loss_value, model_state, state_dict_epoch)

        # Assertions
        self.assertEqual(state_dict_epoch["epoch"], epoch)
        self.assertEqual(state_dict_epoch["loss"], loss_value)
        self.assertEqual(state_dict_epoch["state_dict"], model_state)

    def test_choose_device_mps_available(self):
        # Mock the situation where MPS (Multi-Process Service) is available
        with unittest.mock.patch("torch.backends.mps.is_available", return_value=True):
            device = choose_device()
            self.assertEqual(device.type, "mps")

    def test_choose_device_cuda_available(self):
        # Mock the situation where CUDA is available
        with unittest.mock.patch(
            "torch.backends.mps.is_available", return_value=False
        ), unittest.mock.patch("torch.cuda.is_available", return_value=True):
            device = choose_device()
            self.assertEqual(device.type, "cuda")

    def test_choose_device_cpu_only(self):
        # Mock the situation where neither MPS nor CUDA is available
        with unittest.mock.patch(
            "torch.backends.mps.is_available", return_value=False
        ), unittest.mock.patch("torch.cuda.is_available", return_value=False):
            device = choose_device()
            self.assertEqual(device.type, "cpu")

    def setUp(self):
        # Set up any necessary data or mocks for testing
        pass

    @patch("time.time", return_value=0)
    @patch(
        "train_pipeline.train_pipeline.plot_loss_accuracy",
        new_callable=MagicMock,
    )
    def test_train_function(self, time_mock, plot_loss_accuracy_mock):
        expected_epoch_value = 4
        expected_loss_value = 2.0

        # Mock data and objects
        class MockDataset:
            def __init__(self):
                self.data = torch.randn((100, 3, 32, 32))
                self.targets = torch.randint(0, 10, (100,))

            def __getitem__(self, index):
                return self.data[index], self.targets[index]

            def __len__(self):
                return len(self.data)

        model = vgg_a()  # Instantiate your model
        train_dataset = MockDataset()
        num_epochs = 5  # Adjust as needed
        state_dict_epoch = {"epoch": -1, "loss": float("inf"), "state_dict": None}

        # Mock the update_state_epoch function
        def mock_update_state_epoch(epoch, loss_value, model_state, state_dict_epoch):
            state_dict_epoch["epoch"] = epoch
            state_dict_epoch["loss"] = loss_value
            state_dict_epoch["state_dict"] = model_state

        train(
            model, train_dataset, num_epochs, state_dict_epoch, mock_update_state_epoch
        )

        # Add your assertions based on the expected behavior of your train function
        # For example, you might check if the state_dict_epoch has been updated correctly
        self.assertEqual(state_dict_epoch["epoch"], expected_epoch_value)
        self.assertAlmostEqual(state_dict_epoch["loss"], expected_loss_value, places=0)
        # Add more assertions as needed


if __name__ == "__main__":
    unittest.main()
