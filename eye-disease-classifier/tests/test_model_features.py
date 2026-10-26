import unittest
from pathlib import Path

import numpy as np

from src.predict import image_quality_checks

class FeatureTests(unittest.TestCase):
    def test_quality_checks_report_blur_and_aspect_ratio(self):
        try:
            from PIL import Image
        except ImportError:
            self.skipTest("Pillow is not installed")
        image = Image.new("RGB", (400, 100), color=(128, 128, 128))
        codes = {item["code"] for item in image_quality_checks(image)}
        self.assertIn("aspect_ratio", codes)
        self.assertIn("blur", codes)

    def test_real_checkpoint_gradcam(self):
        checkpoint = Path("artifacts/retinal_classifier.keras")
        if not checkpoint.exists():
            self.skipTest("Saved checkpoint is not available")
        try:
            import tensorflow as tf
            from PIL import Image
            from src.explain import make_gradcam_heatmap
            from src.predict import preprocess_image
        except ImportError:
            self.skipTest("TensorFlow or Pillow is not installed")
        sample = next(Path("data/processed/test").rglob("*.jpg"), None)
        if sample is None:
            self.skipTest("No test image is available")
        model = tf.keras.models.load_model(checkpoint)
        batch = preprocess_image(Image.open(sample).convert("RGB"), 224)
        heatmap = make_gradcam_heatmap(batch, model, class_index=0)
        self.assertEqual(heatmap.ndim, 2)
        self.assertTrue(np.isfinite(heatmap).all())
        self.assertGreaterEqual(float(heatmap.min()), 0.0)
        self.assertLessEqual(float(heatmap.max()), 1.0)

if __name__ == "__main__":
    unittest.main()
