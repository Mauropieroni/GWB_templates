import unittest

from gwb_templates.template import Template, get_template_from_registry


class TestRegistry(unittest.TestCase):

    def test_all_labels_retrievable(self):
        for label in Template.registered_templates():
            with self.subTest(label=label):
                model = get_template_from_registry(label)
                self.assertIsNotNone(model)
                self.assertEqual(model.model_name, label)

    def test_unknown_label_raises(self):
        with self.assertRaises(ValueError):
            get_template_from_registry("does_not_exist")


if __name__ == "__main__":
    unittest.main(verbosity=2)
