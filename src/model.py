"""EfficientNetB0 model factory."""
from __future__ import annotations

def build_model(num_classes: int, image_size: int = 224, fine_tune: bool = False, weights: str | None = "imagenet"):
    import tensorflow as tf
    if num_classes < 2: raise ValueError("num_classes must be at least 2")
    try:
        backbone = tf.keras.applications.EfficientNetB0(include_top=False, weights=weights, input_shape=(image_size, image_size, 3))
    except Exception as exc:
        if weights == "imagenet":
            print(f"ImageNet weights unavailable ({exc}); using random initialization.")
            backbone = tf.keras.applications.EfficientNetB0(include_top=False, weights=None, input_shape=(image_size, image_size, 3))
        else: raise
    backbone.trainable = fine_tune
    if fine_tune:
        for layer in backbone.layers[:-25]: layer.trainable = False
        for layer in backbone.layers:
            if isinstance(layer, tf.keras.layers.BatchNormalization): layer.trainable = False
    inputs = tf.keras.Input(shape=(image_size, image_size, 3), name="eye_image")
    x = backbone(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_average_pool")(x)
    x = tf.keras.layers.Dropout(0.35)(x); x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="class_probabilities")(x)
    model = tf.keras.Model(inputs, outputs, name="retinal_efficientnetb0")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model
