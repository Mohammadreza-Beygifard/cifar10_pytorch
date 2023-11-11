import unittest
import torchvision
from data_pipeline.data_pipeline import get_cifar10

class TestGetCIFAR10(unittest.TestCase):
    def test_get_cifar10(self):
        original_train_dataset, test_dataset = get_cifar10()
        
        # Assert that the original_train_dataset and test_dataset are not None
        self.assertIsNotNone(original_train_dataset)
        self.assertIsNotNone(test_dataset)
        
        # Assert that the original_train_dataset and test_dataset are instances of torchvision.datasets.CIFAR10
        self.assertIsInstance(original_train_dataset, torchvision.datasets.CIFAR10)
        self.assertIsInstance(test_dataset, torchvision.datasets.CIFAR10)
        
        # Add more assertions as needed
        
if __name__ == '__main__':
    unittest.main()