"""Model-derived Grad-CAM utilities."""
from __future__ import annotations
import numpy as np

def _last_conv_layer(model):
    import tensorflow as tf
    candidates = [layer for layer in model.layers if isinstance(layer, tf.keras.layers.Conv2D)]
    for block in (layer for layer in model.layers if isinstance(layer, tf.keras.Model)):
        candidates.extend(layer for layer in block.layers if isinstance(layer, tf.keras.layers.Conv2D))
    if not candidates: raise ValueError("The loaded model has no Conv2D layer suitable for Grad-CAM.")
    return candidates[-1]

def make_gradcam_heatmap(image_batch: np.ndarray, model, class_index: int | None = None) -> np.ndarray:
    import tensorflow as tf
    if image_batch.ndim != 4 or image_batch.shape[0] != 1:
        raise ValueError("Grad-CAM expects a batch containing exactly one image.")
    conv_layer = _last_conv_layer(model)
    nested = next((layer for layer in model.layers if isinstance(layer, tf.keras.Model)
                   and conv_layer in layer.layers), None)
    if nested is not None:
        # A nested EfficientNet layer cannot be connected directly to the outer
        # model input in Keras 3. Extract backbone activations, then replay the
        # saved outer classification head under the same GradientTape.
        feature_model = tf.keras.Model(nested.input, [conv_layer.output, nested.output])
        head_layers = list(model.layers[model.layers.index(nested) + 1:])
    else:
        feature_model = tf.keras.Model(model.inputs, [conv_layer.output, model.output])
        head_layers = []
    with tf.GradientTape() as tape:
        conv_outputs, features = feature_model(image_batch, training=False)
        predictions = features
        for layer in head_layers:
            predictions = layer(predictions, training=False)
        if class_index is None: class_index = int(tf.argmax(predictions[0]))
        if not 0 <= class_index < predictions.shape[-1]: raise ValueError("class_index is outside the model output range")
        score = predictions[:, class_index]
    gradients = tape.gradient(score, conv_outputs)
    if gradients is None: raise RuntimeError("Gradients were unavailable for Grad-CAM.")
    weights = tf.reduce_mean(gradients, axis=(1, 2), keepdims=True)
    heatmap = tf.maximum(tf.reduce_sum(weights * conv_outputs, axis=-1)[0], 0)
    result = tf.math.divide_no_nan(heatmap, tf.reduce_max(heatmap))
    return result.numpy().astype(np.float32)

def overlay_heatmap(image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.38) -> np.ndarray:
    import matplotlib
    from PIL import Image
    base = Image.fromarray(np.asarray(image, dtype=np.uint8)).convert("RGB")
    resized = Image.fromarray(np.uint8(np.clip(heatmap, 0, 1) * 255)).resize(base.size)
    if hasattr(matplotlib, "colormaps"):
        colormap = matplotlib.colormaps["jet"]
    else:
        import matplotlib.pyplot as plt
        colormap = plt.get_cmap("jet")
    colored = (colormap(np.asarray(resized) / 255.0)[..., :3] * 255).astype(np.uint8)
    return np.clip(np.asarray(base) * (1 - alpha) + colored * alpha, 0, 255).astype(np.uint8)
