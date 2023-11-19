import unittest
from unittest.mock import patch
import torchvision
from data_pipeline.data_pipeline import get_cifar10, get_train_data, CustomDataset
import torchvision.transforms as transforms
from data_pipeline.data_pipeline import transform_provider
import torch


class MockCIFAR10Dataset(torch.utils.data.Dataset):
    def __init__(self, root, train, download, transform):
        self.root = root
        self.train = train
        self.download = download
        self.transform = transform

    def __len__(self):
        return 50  # Adjust the length as needed

    def __getitem__(self, idx):
        if idx >= len(self):
            raise IndexError("index out of range")
        return torch.rand(3, 32, 32), torch.randint(0, 10, (1,)).item()


class TestGetCIFAR10(unittest.TestCase):
    def test_primary_transform(self):
        transform_type = "primary"
        expected_transform = transforms.Compose(
            [
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )
        actual_transform = transform_provider(transform_type)

        expected_transform_list = [t.__dict__ for t in expected_transform.transforms]
        actual_transform_list = [t.__dict__ for t in actual_transform.transforms]

        self.assertEqual(actual_transform_list, expected_transform_list)

    def test_test_transform(self):
        transform_type = "test"
        expected_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )
        actual_transform = transform_provider(transform_type)

        expected_transform_list = [t.__dict__ for t in expected_transform.transforms]
        actual_transform_list = [t.__dict__ for t in actual_transform.transforms]

        self.assertEqual(actual_transform_list, expected_transform_list)

    def test_train_transform(self):
        transform_type = "train"
        expected_transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.RandomCrop(32, padding=4),
                transforms.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
                ),
                transforms.RandomResizedCrop(
                    32, scale=(0.8, 1.0), ratio=(0.9, 1.1), antialias=True
                ),
                transforms.GaussianBlur(kernel_size=3),
            ]
        )
        actual_transform = transform_provider(transform_type)

        expected_transform_list = [t.__dict__ for t in expected_transform.transforms]
        actual_transform_list = [t.__dict__ for t in actual_transform.transforms]

        self.assertEqual(actual_transform_list, expected_transform_list)

    def test_invalid_transform(self):
        transform_type = "invalid"
        with self.assertRaises(ValueError):
            transform_provider(transform_type)

    def test_get_cifar10(self):
        original_dataset, test_dataset = get_cifar10()

        # Assert that the original_train_dataset and test_dataset are not None
        self.assertIsNotNone(original_dataset)
        self.assertIsNotNone(test_dataset)

        # Assert that the original_train_dataset and test_dataset are instances of torchvision.datasets.CIFAR10
        self.assertIsInstance(original_dataset, torchvision.datasets.CIFAR10)
        self.assertIsInstance(test_dataset, torchvision.datasets.CIFAR10)

    @patch("data_pipeline.data_pipeline.get_cifar10", autospec=True)
    @patch("data_pipeline.data_pipeline.transform_provider", autospec=True)
    def test_get_train_data(self, transform_provider_mock, get_cifar10_mock):
        # Create a MagicMock for the original dataset
        original_dataset_mock = MockCIFAR10Dataset(
            root="./data",
            train=True,
            download=True,
            transform=transforms.ToTensor(),  # You can provide an appropriate transform here
        )

        # Set the return value for get_cifar10_mock
        get_cifar10_mock.return_value = (original_dataset_mock, None)

        # Call the function under test
        train_dataset = get_train_data()

        # Assertions
        self.assertEqual(
            len(train_dataset), 2 * len(original_dataset_mock)
        )  # Length should be twice the length of the original dataset

        # Ensure that transform_provider is called with the correct arguments
        transform_provider_mock.assert_any_call("primary")
        transform_provider_mock.assert_any_call("train")


if __name__ == "__main__":
    unittest.main()
