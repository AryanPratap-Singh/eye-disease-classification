"""Streamlit interface for the eye disease classification project."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import streamlit as st

from src.common import METADATA_PATH, MODEL_PATH, REPORTS
from src.predict import image_quality_checks, preprocess_image, validate_image

st.set_page_config(
    page_title="Eye Disease Classification",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """<style>
    :root {
        --app-bg: #f4f7f9;
        --panel-bg: #ffffff;
        --soft-bg: #f8fafc;
        --sidebar-bg: #eaf1f4;
        --text-main: #102a43;
        --text-muted: #52606d;
        --border: #cbd5e1;
        --accent: #1f6f78;
    }
    [data-theme="dark"] {
        --app-bg: #101820;
        --panel-bg: #182631;
        --soft-bg: #202f3b;
        --sidebar-bg: #121f29;
        --text-main: #e6edf3;
        --text-muted: #b5c2cc;
        --border: #405363;
        --accent: #67c5c4;
    }
    @media (prefers-color-scheme: dark) {
        :root {
            --app-bg: #101820;
            --panel-bg: #182631;
            --soft-bg: #202f3b;
            --sidebar-bg: #121f29;
            --text-main: #e6edf3;
            --text-muted: #b5c2cc;
            --border: #405363;
            --accent: #67c5c4;
        }
    }
    .stApp { background: var(--app-bg); color: var(--text-main); }
    [data-testid="stSidebar"] { background: var(--sidebar-bg); border-right: 1px solid var(--border); }
    .block-container { max-width: 1180px; padding-top: 2rem; }
    .project-kicker { color: var(--accent); font-size: .78rem; font-weight: 700; letter-spacing: .08em; text-transform: uppercase; }
    .project-title { color: var(--text-main); font-size: 2.1rem; font-weight: 700; margin: .15rem 0 .25rem; }
    .project-subtitle { color: var(--text-muted); margin-bottom: 1.5rem; }
    .result-box { border: 1px solid var(--border); border-radius: 12px; padding: 1rem 1.2rem; background: var(--soft-bg); color: var(--text-main); }
    .description-box { border-left: 4px solid var(--accent); border-radius: 8px; padding: 1.1rem 1.25rem; background: var(--panel-bg); color: var(--text-main); font-size: 1.08rem; line-height: 1.65; }
    </style>""",
    unsafe_allow_html=True,
)
st.markdown('<div class="project-kicker">Academic deep learning project</div>', unsafe_allow_html=True)
st.markdown('<div class="project-title">Eye Disease Classification</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="project-subtitle">Upload an eye image to view the model result and an optional visual explanation.</div>',
    unsafe_allow_html=True,
)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


@st.cache_resource
def load_saved():
    if not MODEL_PATH.exists() or not METADATA_PATH.exists():
        return None, None
    import tensorflow as tf

    metadata = read_json(METADATA_PATH)
    model = tf.keras.models.load_model(MODEL_PATH)
    class_names = metadata.get("class_names", [])
    if int(model.output_shape[-1]) != len(class_names):
        raise ValueError("Saved model output does not match metadata.json.")
    return model, metadata


DISEASE_INFO = {
    "Cataract": (
        "A cataract is clouding of the eye's natural lens. It can cause blurry vision, glare, "
        "and reduced clarity. Only an eye-care professional can confirm a cataract diagnosis."
    ),
    "Conjunctivitis": (
        "Conjunctivitis is inflammation of the conjunctiva, the thin tissue covering the white "
        "of the eye. Redness, irritation, watering, and discharge may occur. Its cause can vary."
    ),
    "Eyelid": (
        "Eyelid is a dataset label for images showing an eyelid-related condition or appearance. "
        "The label is broad and does not identify a specific medical diagnosis."
    ),
    "Normal": (
        "Normal is the dataset's non-condition label. A model score for this class does not prove "
        "that an eye is healthy and cannot replace an eye examination."
    ),
    "Uveitis": (
        "Uveitis refers to inflammation inside the eye involving the uveal tract. It may cause "
        "pain, redness, light sensitivity, or blurred vision and requires professional assessment."
    ),
}

with st.sidebar:
    st.header("Project navigation")
    page = st.radio(
        "Choose a page",
        ["Overview", "Classify an Image", "Evaluation & Graphs", "About and Limitations"],
    )
    st.divider()
    st.caption("EfficientNetB0 · 224 × 224 RGB")

try:
    model, metadata = load_saved()
    load_error = None
except Exception as exc:
    model, metadata, load_error = None, None, str(exc)

default_classes = ["Cataract", "Conjunctivitis", "Eyelid", "Normal", "Uveitis"]

if page == "Overview":
    st.subheader("Welcome")
    st.write(
        "This application classifies an uploaded eye image into one of the labelled classes in the "
        "dataset. Use Classify an Image for the main demonstration, then use Evaluation & Graphs to "
        "review the saved test results."
    )
    classes = (metadata or {}).get("class_names", default_classes)
    metrics = read_json(REPORTS / "metrics.json")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Supported classes", len(classes))
    c2.metric("Input size", f"{(metadata or {}).get('image_size', 224)} × {(metadata or {}).get('image_size', 224)}")
    c3.metric("Held-out images", metrics.get("n_test_images", "Unavailable") if metrics else "Unavailable")
    c4.metric("Test accuracy", f"{metrics['accuracy']:.1%}" if metrics else "Unavailable")
    st.markdown("### Supported classes")
    st.write(" · ".join(classes))
    if load_error:
        st.error(f"The saved model could not be loaded: {load_error}")
    elif model is None:
        st.info("No compatible saved model is available.")

elif page == "Classify an Image":
    st.subheader("Classify an Image")
    st.write("Upload one JPG, JPEG, or PNG image to view the model's predicted class and score.")
    uploaded = st.file_uploader(
        "Choose an eye image",
        type=["jpg", "jpeg", "png"],
        help="Maximum file size: 10 MB.",
    )
    if uploaded is None:
        st.info("Your prediction will appear here after you upload an image.")
    elif model is None or metadata is None:
        st.warning("A compatible saved model and metadata file are required.")
    else:
        try:
            image = validate_image(uploaded)
            warnings = image_quality_checks(image)
            image_size = int(metadata["image_size"])
            batch = preprocess_image(image, image_size)
            with st.spinner("Analyzing image..."):
                scores = np.asarray(model.predict(batch, verbose=0))[0]
            class_names = list(metadata["class_names"])
            top_index = int(np.argmax(scores))
            predicted_class = class_names[top_index]
            confidence = float(scores[top_index])

            left, right = st.columns([1, 1.15], gap="large")
            with left:
                st.image(image, caption=f"Uploaded image · {image.width} × {image.height}px", use_container_width=True)
                if warnings:
                    st.warning("Image-quality warnings")
                    for warning in warnings:
                        st.write(f"- {warning['message']}")
                else:
                    st.success("Basic image-quality checks passed.")
            with right:
                st.markdown("#### Model result")
                st.markdown(
                    f'<div class="result-box"><small>Predicted class</small><h2>{predicted_class}</h2>'
                    f'<small>Top model score</small><h3>{confidence:.1%}</h3></div>',
                    unsafe_allow_html=True,
                )
                st.caption("Score produced by the trained classification model.")
                ranked = sorted(zip(class_names, scores.tolist()), key=lambda item: item[1], reverse=True)
                st.markdown("#### Scores for all classes")
                st.bar_chart(
                    {"model score": {name: float(score) for name, score in ranked}},
                    horizontal=True,
                )

            st.divider()
            st.markdown(f"#### About {predicted_class}")
            st.markdown(
                f'<div class="description-box">{DISEASE_INFO.get(predicted_class, "No description is available for this label.")}</div>',
                unsafe_allow_html=True,
            )

            with st.expander("Optional: view Grad-CAM explanation"):
                st.write(
                    "Grad-CAM highlights image regions that influenced the selected class score."
                )
                selected_name = st.selectbox(
                    "Class to explain",
                    class_names,
                    index=top_index,
                    key="gradcam_class",
                )
                if st.button("Generate Grad-CAM", type="primary"):
                    try:
                        from src.explain import make_gradcam_heatmap, overlay_heatmap

                        heatmap = make_gradcam_heatmap(batch, model, class_names.index(selected_name))
                        resized = np.asarray(image.resize((image_size, image_size)))
                        overlay = overlay_heatmap(resized, heatmap)
                        original_col, heatmap_col = st.columns(2)
                        original_col.image(resized, caption="Model input", use_container_width=True)
                        heatmap_col.image(
                            overlay,
                            caption=f"Grad-CAM overlay · {selected_name}",
                            use_container_width=True,
                        )
                    except Exception as exc:
                        st.error(f"Grad-CAM could not be generated: {exc}")
        except Exception as exc:
            st.error(str(exc))

elif page == "Evaluation & Graphs":
    st.subheader("Evaluation & Graphs")
    metrics = read_json(REPORTS / "metrics.json")
    report = read_json(REPORTS / "classification_report.json")
    if not metrics:
        st.info("No real evaluation report is available yet.")
    else:
        st.caption("These charts use the saved held-out test results. No placeholder metrics are shown.")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy", f"{metrics['accuracy']:.1%}")
        c2.metric("Balanced accuracy", f"{metrics['balanced_accuracy']:.1%}")
        c3.metric("Macro F1", f"{metrics['macro_f1']:.1%}")
        c4.metric("Weighted F1", f"{metrics['weighted_f1']:.1%}")
        if report:
            class_names = metrics["class_names"]
            per_class = {
                name: {
                    "Precision": float(report[name]["precision"]),
                    "Recall": float(report[name]["recall"]),
                    "F1-score": float(report[name]["f1-score"]),
                }
                for name in class_names
                if name in report
            }
            st.markdown("### Per-class performance")
            st.bar_chart(per_class)
            st.dataframe(per_class, use_container_width=True)
        for filename in ("confusion_matrix.png", "training_curves.png"):
            report_path = REPORTS / filename
            if report_path.exists():
                st.image(str(report_path), caption=filename, use_container_width=True)

else:
    st.subheader("About and Limitations")
    st.markdown(
        "**Architecture:** ImageNet-pretrained EfficientNetB0 with global average pooling, dropout, "
        "and a dynamically sized softmax head.\n\n"
        "**Dataset limitation:** the supplied images appear to be external eye-condition photographs, "
        "not verified retinal fundus photographs. Results must not be presented as clinically validated "
        "retinal disease detection.\n\n"
        "**Responsible use:** this is an undergraduate academic demonstration. It has no clinical "
        "validation, calibration, external validation, or diagnostic authority."
    )
