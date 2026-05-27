import unittest

from gwb_templates.templates import _REGISTRY, get_template


class TestRegistry(unittest.TestCase):

    def test_all_labels_retrievable(self):
        for label in _REGISTRY:
            with self.subTest(label=label):
                model = get_template(label)
                self.assertIsNotNone(model)
                self.assertEqual(model.model_name, label)

    def test_unknown_label_raises(self):
        with self.assertRaises(ValueError):
            get_template("does_not_exist")


if __name__ == "__main__":
    unittest.main(verbosity=2)
