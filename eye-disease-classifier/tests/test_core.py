import unittest
from pathlib import Path

from src.data import discover_classes, validate_split
from src.evaluate import calculate_metrics

class CoreTests(unittest.TestCase):
    def test_class_discovery_is_sorted_and_validation_counts(self):
        project_root = Path(__file__).resolve().parents[1]
        root = project_root / "data" / "raw" / "Image Dataset on Eye Diseases Classification (Uveitis, Conjunctivitis, Cataract, Eyelid) with Symptoms and SMOTE Validation"
        if not root.is_dir():
            root = project_root.parent / "Image Dataset on Eye Diseases Classification (Uveitis, Conjunctivitis, Cataract, Eyelid) with Symptoms and SMOTE Validation"
        if not root.is_dir():
            self.skipTest("Supplied dataset is not available")
        self.assertEqual(discover_classes(root), ["Cataract", "Conjunctivitis", "Eyelid", "Normal", "Uveitis"])
        self.assertEqual(validate_split(root)["Normal"], 649)

    def test_missing_split_is_clear(self):
        with self.assertRaises(FileNotFoundError):
            discover_classes("does-not-exist")

    def test_metrics_include_ordered_classes(self):
        try:
            result = calculate_metrics([0, 1, 2, 2], [0, 1, 1, 2], ["A", "B", "C"])
        except ImportError:
            self.skipTest("scikit-learn is not installed")
        self.assertEqual(result["class_names"], ["A", "B", "C"])
        self.assertIn("macro_precision", result)
        self.assertEqual(result["n_test_images"], 4)

if __name__ == "__main__":
    unittest.main()
