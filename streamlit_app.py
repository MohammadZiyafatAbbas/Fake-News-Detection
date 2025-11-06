"""Streamlit UI for the Fake News Detection project.

This lightweight wrapper loads the project's classifier (or a safe fallback)
and exposes a small web UI suitable for deployment to Streamlit Sharing or
Streamlit for Teams.
"""
import streamlit as st
from pathlib import Path
import joblib
import numpy as _np
from src.config import MODEL_SAVE_PATH
from src.models.classifier import FakeNewsClassifier
from src.utils.data_utils import load_dataset
import logging

st.set_page_config(page_title="Fake News Detector", layout="centered")
logger = logging.getLogger(__name__)


@st.cache_resource
def get_classifier():
    # Try to load persisted model; otherwise fall back to a safe rule-based
    # classifier similar to the Flask app behavior.
    try:
        clf = FakeNewsClassifier.load_model(str(MODEL_SAVE_PATH))
        logger.info("Loaded persisted model for Streamlit app.")
        return clf
    except Exception as e:
        logger.warning(f"Failed to load persisted model: {e}. Using fallback.")

        # Build fallback classifier (rule-based) wrapped in FakeNewsClassifier
        class _Fallback:
            def predict(self, X):
                return _np.array([0 if 'fake' in str(x).lower() or 'hoax' in str(x).lower() else 1 for x in X])

            def predict_proba(self, X):
                probs = []
                for x in X:
                    if 'fake' in str(x).lower() or 'hoax' in str(x).lower():
                        probs.append([0.9, 0.1])
                    else:
                        probs.append([0.1, 0.9])
                return _np.array(probs)

        demo = FakeNewsClassifier()
        demo.feature_pipeline = None
        demo.best_model = _Fallback()
        return demo


def main():
    st.title("Fake News Detector")
    st.write("Paste an article or a paragraph and press Analyze. This demo uses a fallback classifier if no trained model is available.")

    text = st.text_area("Article text", height=250)
    if st.button("Analyze"):
        if not text or not text.strip():
            st.error("Please enter some text to analyze.")
            return

        clf = get_classifier()

        try:
            # Some classifiers expect a feature pipeline, but fallback ignores it.
            prediction, probability = clf.predict(text)
        except Exception:
            # If classifier expects pipeline.transform, try to call it.
            try:
                feat = clf.feature_pipeline.transform([text]) if getattr(clf, 'feature_pipeline', None) is not None else [text]
                prediction = clf.best_model.predict(feat)[0]
                try:
                    probability = clf.best_model.predict_proba(feat)[0][1]
                except Exception:
                    probability = 0.5
            except Exception as ex:
                st.error(f"Prediction failed: {ex}")
                return

        label = "FAKE" if prediction == 0 else "REAL"
        score = round(float(probability) * 100, 2)

        if prediction == 0:
            st.error(f"Prediction: {label} — Confidence: {score}%")
        else:
            st.success(f"Prediction: {label} — Confidence: {score}%")

        with st.expander("Show raw output"):
            st.json({"prediction": int(prediction), "probability": float(probability)})


if __name__ == "__main__":
    main()
