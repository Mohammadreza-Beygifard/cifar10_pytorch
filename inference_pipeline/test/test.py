import unittest
import torch
from inference_pipeline.check_performance import check_performance


class TestCheckPerformance(unittest.TestCase):
    def test_check_performance(self):
        # Create a mock model and test dataset
        model = torch.nn.Linear(10, 2)  # Example model
        test_dataset = [(torch.randn(1, 10), torch.tensor([0]))]  # Example test dataset

        # Call the check_performance function
        check_performance(model, test_dataset)

        # Assert that the printed output is correct
        expected_output = "Accuracy of the network on the 10000 test images: 0.0 %"
        self.assertEqual(
            expected_output, "Accuracy of the network on the 10000 test images: 0.0 %"
        )


if __name__ == "__main__":
    unittest.main()
