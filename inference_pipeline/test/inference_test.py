import unittest
import torch

from inference_pipeline.inference import preprocess_image


class TestPreprocessImage(unittest.TestCase):
    def test_preprocess_image(self):
        image_path = "inference_pipeline/test/test_ship.jpeg"

        result = preprocess_image(image_path)

        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.size(), (1, 3, 32, 32))


if __name__ == "__main__":
    unittest.main()
