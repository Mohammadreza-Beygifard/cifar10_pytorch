import unittest
from unittest.mock import mock_open, patch
from train_pipeline.parameter_provider import (
    get_parameters,
    get_optimizer_parameters,
    get_scheduler_parameters,
    get_training_parameters,
)


class TestTrainingParameters(unittest.TestCase):
    @patch(
        "builtins.open",
        mock_open(
            read_data='{"optimizer": {"learning_rate": 0.001, "weight_decay": 0.0001}, "scheduler": {"active": true, "gamma": 0.9}, "batch_size": 128, "num_epochs": 100}'
        ),
    )
    def test_get_parameters(self):
        result = get_parameters()
        self.assertEqual(result["optimizer"]["learning_rate"], 0.001)
        self.assertEqual(result["optimizer"]["weight_decay"], 0.0001)
        self.assertTrue(result["scheduler"]["active"])
        self.assertEqual(result["scheduler"]["gamma"], 0.9)
        self.assertEqual(result["batch_size"], 128)
        self.assertEqual(result["num_epochs"], 100)

    def test_get_optimizer_parameters(self):
        with patch(
            "train_pipeline.parameter_provider.get_parameters",
            return_value={
                "optimizer": {"learning_rate": 0.001, "weight_decay": 0.0001}
            },
        ):
            result = get_optimizer_parameters()
            self.assertEqual(result.learning_rate, 0.001)
            self.assertEqual(result.weight_decay, 0.0001)

    def test_get_scheduler_parameters(self):
        with patch(
            "train_pipeline.parameter_provider.get_parameters",
            return_value={"scheduler": {"active": True, "gamma": 0.9}},
        ):
            result = get_scheduler_parameters()
            self.assertTrue(result.active)
            self.assertEqual(result.gamma, 0.9)

    def test_get_training_parameters(self):
        with patch(
            "train_pipeline.parameter_provider.get_parameters",
            return_value={
                "optimizer": {"learning_rate": 0.001, "weight_decay": 0.0001},
                "scheduler": {"active": True, "gamma": 0.9},
                "batch_size": 128,
                "num_epochs": 100,
            },
        ):
            result = get_training_parameters()
            self.assertEqual(result.optimizer_parameters.learning_rate, 0.001)
            self.assertEqual(result.optimizer_parameters.weight_decay, 0.0001)
            self.assertTrue(result.scheduler_parameters.active)
            self.assertEqual(result.scheduler_parameters.gamma, 0.9)
            self.assertEqual(result.batch_size, 128)
            self.assertEqual(result.num_epochs, 100)


if __name__ == "__main__":
    unittest.main()
