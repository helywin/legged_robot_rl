import unittest
from typing import get_type_hints

import torch
from torch import nn

from examples.pytorch_module_prediction import OneInputQModule


class PyTorchModulePredictionTest(unittest.TestCase):
    def test_model_is_nn_module(self) -> None:
        model = OneInputQModule(weight=3.0, bias=0.1)

        self.assertIsInstance(model, nn.Module)

    def test_parameters_are_registered_by_name(self) -> None:
        model = OneInputQModule(weight=3.0, bias=0.1)

        parameters = dict(model.named_parameters())

        self.assertEqual(set(parameters), {"weight", "bias"})
        self.assertAlmostEqual(parameters["weight"].item(), 3.0)
        self.assertAlmostEqual(parameters["bias"].item(), 0.1)

    def test_calling_module_runs_forward_prediction(self) -> None:
        model = OneInputQModule(weight=3.0, bias=0.1)

        prediction = model(torch.tensor(-0.02))

        self.assertIsInstance(prediction, torch.Tensor)
        self.assertAlmostEqual(prediction.item(), 0.04)

    def test_call_signature_exposes_tensor_type_to_ide(self) -> None:
        hints = get_type_hints(OneInputQModule.__call__)

        self.assertIs(hints["observation"], torch.Tensor)
        self.assertIs(hints["return"], torch.Tensor)

    def test_typed_call_keeps_module_forward_hooks(self) -> None:
        model = OneInputQModule(weight=3.0, bias=0.1)
        hook_outputs: list[torch.Tensor] = []
        hook = model.register_forward_hook(
            lambda _module, _inputs, output: hook_outputs.append(output)
        )

        try:
            prediction = model(torch.tensor(-0.02))
        finally:
            hook.remove()

        self.assertEqual(hook_outputs, [prediction])

    def test_forward_does_not_change_parameters(self) -> None:
        model = OneInputQModule(weight=3.0, bias=0.1)
        before = [parameter.detach().clone() for parameter in model.parameters()]

        model(torch.tensor(-0.02))

        self.assertTrue(
            all(
                torch.equal(old, current)
                for old, current in zip(before, model.parameters())
            )
        )


if __name__ == "__main__":
    unittest.main()
