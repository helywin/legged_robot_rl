import unittest

from examples.python_class_model import OneInputQModel


class PythonClassModelTest(unittest.TestCase):
    def test_constructor_stores_parameters(self) -> None:
        model = OneInputQModel(weight=3.0, bias=0.1)

        self.assertEqual(model.weight, 3.0)
        self.assertEqual(model.bias, 0.1)

    def test_predict_uses_stored_parameters(self) -> None:
        model = OneInputQModel(weight=3.0, bias=0.1)

        self.assertAlmostEqual(model.predict(-0.02), 0.04)
        self.assertAlmostEqual(model.predict(0.05), 0.25)

    def test_two_objects_keep_independent_parameters(self) -> None:
        first = OneInputQModel(weight=3.0, bias=0.1)
        second = OneInputQModel(weight=-2.0, bias=0.5)

        self.assertAlmostEqual(first.predict(0.2), 0.7)
        self.assertAlmostEqual(second.predict(0.2), 0.1)


if __name__ == "__main__":
    unittest.main()
