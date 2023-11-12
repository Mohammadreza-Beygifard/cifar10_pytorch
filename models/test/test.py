import unittest
import torch
from models.vgg import vgg_a


class TestVGG(unittest.TestCase):
    def test_forward_pass(self):
        # Create an instance of the VGG model
        model = vgg_a()

        # Define dummy input
        dummy_input = torch.randn(1, 3, 32, 32)

        # Perform a forward pass
        output = model(dummy_input)

        # Check if the output has the correct shape
        self.assertEqual(output.shape, torch.Size([1, 10]))


if __name__ == "__main__":
    unittest.main()
