"""Reusable single-image prediction helpers."""
from __future__ import annotations
import numpy as np

def validate_image(uploaded, max_bytes: int = 10 * 1024 * 1024):
    from PIL import Image, ImageOps
    if uploaded is None: raise ValueError("No image was supplied.")
    if getattr(uploaded, "size", 0) > max_bytes: raise ValueError("Image is larger than the 10 MB limit.")
    try:
        image = ImageOps.exif_transpose(Image.open(uploaded)).convert("RGB")
    except Exception as exc: raise ValueError(f"The image is unreadable: {exc}") from exc
    if image.width < 32 or image.height < 32: raise ValueError("Image dimensions are too small (minimum 32x32 pixels).")
    return image

def preprocess_image(image, image_size: int = 224) -> np.ndarray:
    return np.asarray(image.resize((image_size, image_size)), dtype=np.float32)[None, ...]

def image_quality_checks(image, *, min_width: int = 64, min_height: int = 64,
                         blur_threshold: float = 20.0) -> list[dict[str, str]]:
    """Return non-fatal quality warnings for a readable RGB PIL image."""
    warnings: list[dict[str, str]] = []
    width, height = image.size
    if width < min_width or height < min_height:
        warnings.append({"code": "small", "message": f"The image is small ({width} × {height}); fine details may be unreliable."})
    ratio = width / max(height, 1)
    if ratio < 0.5 or ratio > 2.0:
        warnings.append({"code": "aspect_ratio", "message": "The image has an unusual aspect ratio and may not be a typical eye photograph."})
    pixels = np.asarray(image.convert("L"), dtype=np.float32)
    brightness = float(pixels.mean())
    if brightness < 25:
        warnings.append({"code": "dark", "message": "The image is unusually dark; image content may be difficult for the model to use."})
    elif brightness > 230:
        warnings.append({"code": "bright", "message": "The image is unusually bright; overexposure may reduce useful detail."})
    if pixels.shape[0] > 2 and pixels.shape[1] > 2:
        laplacian = (-4 * pixels[1:-1, 1:-1] + pixels[:-2, 1:-1] + pixels[2:, 1:-1] +
                     pixels[1:-1, :-2] + pixels[1:-1, 2:])
        if float(laplacian.var()) < blur_threshold:
            warnings.append({"code": "blur", "message": "The image may be blurry or out of focus; prediction confidence may be less reliable."})
    return warnings

def predict_image(model, image, class_names: list[str], image_size: int = 224) -> tuple[str, np.ndarray]:
    probabilities = np.asarray(model.predict(preprocess_image(image, image_size), verbose=0))[0]
    if probabilities.shape != (len(class_names),): raise ValueError("Model output does not match saved class mapping.")
    return class_names[int(np.argmax(probabilities))], probabilities
