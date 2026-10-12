"""
Streamlit Web Application: Skin Cancer Classification Using CNN.

Features:
- Image upload (JPG, JPEG, PNG)
- Real-time preview and preprocessing
- Multi-class prediction with ranked confidence breakdown
- Prominent educational and medical disclaimers
- Graceful missing-model handling with step-by-step training guidance
- Interactive knowledge base on the 7 HAM10000 lesion categories
"""

import sys
from pathlib import Path
from typing import Optional

try:
    import streamlit as st
    import numpy as np
    import pandas as pd
    from PIL import Image
except ImportError as e:
    print("=" * 72)
    print("  [DEPENDENCY NOTICE] Required Python Libraries Not Found")
    print("=" * 72)
    print(f"Missing package error: {e}")
    print("\nPlease activate your virtual environment and install dependencies:")
    print("  .\\venv\\Scripts\\Activate.ps1")
    print("  pip install -r requirements.txt")
    print("\nThen run the Streamlit app:")
    print("  streamlit run app.py")
    print("=" * 72)
    sys.exit(1)

# Import project utilities
from src.dataset import LESION_CLASSES, CLASS_NAMES, preprocess_single_image

# Page configuration
st.set_page_config(
    page_title="Skin Cancer Classification CNN",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
MODEL_PATH = Path("models/skin_cancer_cnn.keras")
DEFAULT_IMG_SIZE = (128, 128)


@st.cache_resource(show_spinner=False)
def load_trained_model(model_path: Path):
    """Loads and caches the trained Keras model."""
    import tensorflow as tf
    if not model_path.exists():
        return None
    return tf.keras.models.load_model(str(model_path))


def render_sidebar():
    st.sidebar.title("🔬 Project Navigation")
    st.sidebar.markdown(
        """
        **Academic Project**:
        Skin Cancer Classification Using Convolutional Neural Networks (CNN) & Python
        
        *Author*: Akshat Pratap  
        *Technology*: TensorFlow, Keras, Streamlit  
        *Dataset*: HAM10000 (10,015 images)
        """
    )
    st.sidebar.divider()

    st.sidebar.subheader("Clinical Disclaimer")
    st.sidebar.warning(
        "⚠️ **Educational Demonstration Only**\n\n"
        "This tool is built solely for educational and academic computer science purposes. "
        "It is **NOT** a medical diagnostic tool. Never rely on automated models for healthcare decisions."
    )
    st.sidebar.divider()

    st.sidebar.subheader("Quick Links")
    st.sidebar.markdown(
        """
        - [HAM10000 Dataset (Kaggle)](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)
        - [Harvard Dataverse Archive](https://doi.org/10.7910/DVN/DBW86T)
        - [ISIC Archive](https://www.isic-archive.com/)
        """
    )


def main():
    render_sidebar()

    # Main Header
    st.title("🔬 Skin Lesion Classification Dashboard")
    st.markdown(
        """
        Welcome to the **Skin Cancer Classification System** powered by a Deep Convolutional Neural Network (CNN).
        This application analyzes dermatoscopic skin lesion images and classifies them into one of 7 standardized clinical categories.
        """
    )

    # Top Tabs
    tab_predict, tab_classes, tab_architecture, tab_disclaimer = st.tabs([
        "🔍 Lesion Classifier",
        "📚 Lesion Categories (HAM10000)",
        "🧠 CNN Architecture",
        "⚖️ Medical & Ethical Disclaimer"
    ])

    # -------------------------------------------------------------
    # TAB 1: CLASSIFIER
    # -------------------------------------------------------------
    with tab_predict:
        # Check model presence
        if not MODEL_PATH.exists():
            st.warning(
                """
                ### ⚠️ Trained Model Not Found!
                The model weights file (`models/skin_cancer_cnn.keras`) has not been generated yet.
                
                **To train your model and enable predictions:**
                1. Ensure your HAM10000 dataset is placed in `data/raw/`.
                2. Open your terminal / PowerShell in the project directory.
                3. Run the training command:
                ```powershell
                python train.py --epochs 15 --batch-size 32
                ```
                4. Refresh this page once training finishes.
                """
            )
            model = None
        else:
            with st.spinner("Loading deep learning model into memory..."):
                model = load_trained_model(MODEL_PATH)

        col1, col2 = st.columns([1, 1], gap="large")

        with col1:
            st.subheader("1. Upload Dermatoscopic Image")
            uploaded_file = st.file_uploader(
                "Choose a skin lesion photograph (.jpg, .jpeg, .png):",
                type=["jpg", "jpeg", "png"],
                help="Upload a clear, well-lit dermatoscopic or close-up image."
            )

            if uploaded_file is not None:
                try:
                    image = Image.open(uploaded_file)
                    st.image(image, caption="Uploaded Lesion Preview", use_column_width=True)
                    st.info(f"Image Resolution: {image.size[0]} x {image.size[1]} pixels | Mode: {image.mode}")
                except Exception as e:
                    st.error(f"Error opening image file: {e}")
                    uploaded_file = None

        with col2:
            st.subheader("2. Model Classification & Confidence")

            if uploaded_file is None:
                st.write("👉 *Please upload an image in the left panel to begin analysis.*")
            elif model is None:
                st.info("Model is currently offline. Please follow the instructions above to train the model first.")
            else:
                predict_button = st.button("🚀 Run CNN Classification", type="primary", use_container_width=True)

                if predict_button:
                    with st.spinner("Analyzing dermatoscopic features with CNN..."):
                        # Preprocess image
                        input_tensor = preprocess_single_image(image, target_size=DEFAULT_IMG_SIZE)
                        probabilities = model.predict(input_tensor, verbose=0)[0]

                        top_idx = int(np.argmax(probabilities))
                        top_code = CLASS_NAMES[top_idx]
                        top_confidence = probabilities[top_idx] * 100.0
                        top_info = LESION_CLASSES[top_code]

                    # Results Banner
                    if top_info["risk_level"].startswith("Critical"):
                        box_type = st.error
                    elif top_info["risk_level"].startswith("High"):
                        box_type = st.warning
                    else:
                        box_type = st.success

                    box_type(
                        f"### Top Predicted Class: **{top_info['full_name']}** (`{top_code}`)\n"
                        f"- **Clinical Category**: {top_info['category']}\n"
                        f"- **Risk Classification**: **{top_info['risk_level']}**\n"
                        f"- **Model Output Confidence**: **{top_confidence:.2f}%**"
                    )

                    st.markdown(
                        f"""
                        > ℹ️ **Clinical Note**: {top_info['clinical_note']}
                        """
                    )

                    # Important Statistical Nuance Warning
                    st.warning(
                        "⚠️ **Important Note on Confidence**: A model confidence score of e.g. 85% "
                        "represents the mathematical similarity of visual features to the training dataset. "
                        "**It is NOT the probability that the patient has cancer.** Deep neural networks "
                        "can exhibit overconfidence even when making incorrect predictions."
                    )

                    # Confidence Scores Breakdown
                    st.subheader("Confidence Distribution Across All 7 Classes")

                    results_data = []
                    for code in CLASS_NAMES:
                        idx = CLASS_NAMES.index(code)
                        prob = float(probabilities[idx]) * 100.0
                        results_data.append({
                            "Code": code,
                            "Category": LESION_CLASSES[code]["full_name"],
                            "Type": LESION_CLASSES[code]["category"],
                            "Confidence (%)": round(prob, 2)
                        })

                    df_results = pd.DataFrame(results_data).sort_values("Confidence (%)", ascending=False)

                    # Display Progress Bars
                    for _, row in df_results.iterrows():
                        col_label, col_bar = st.columns([3, 7])
                        with col_label:
                            st.write(f"**{row['Category']}** ({row['Code']})")
                        with col_bar:
                            st.progress(float(row["Confidence (%)"]) / 100.0, text=f"{row['Confidence (%)']:.2f}%")

                    # Detailed Table
                    with st.expander("📊 View Detailed Probability Table"):
                        st.dataframe(df_results, use_container_width=True)

    # -------------------------------------------------------------
    # TAB 2: LESION CATEGORIES
    # -------------------------------------------------------------
    with tab_classes:
        st.subheader("📚 The 7 Dermatoscopic Lesion Classes in HAM10000")
        st.markdown(
            """
            The HAM10000 dataset collects 10,015 dermatoscopic images annotated into 7 diagnostic categories.
            Understanding the distinction between benign, pre-cancerous, and malignant conditions is critical.
            """
        )

        for code, info in LESION_CLASSES.items():
            with st.expander(f"**{info['full_name']}** (`{code}`) — {info['risk_level']}"):
                st.write(f"**Classification Category:** {info['category']}")
                st.write(f"**Description & Clinical Significance:** {info['clinical_note']}")
                st.write(f"**Risk Level:** `{info['risk_level']}`")

    # -------------------------------------------------------------
    # TAB 3: CNN ARCHITECTURE
    # -------------------------------------------------------------
    with tab_architecture:
        st.subheader("🧠 Deep Learning Architecture & Pipeline")
        st.markdown(
            """
            This academic project utilizes a custom **Convolutional Neural Network (CNN)** designed to balance
            feature extraction capability with lightweight CPU execution on standard student laptops.
            
            ### Pipeline Highlights:
            1. **Patient-Level Data Splitting**:
               Images are partitioned based on `lesion_id` rather than image index. This completely eliminates
               data leakage between the training, validation, and test splits.
            2. **Class Imbalance Mitigation**:
               Because common moles (`nv`) represent >66% of the dataset, Scikit-Learn balanced class weights
               are injected into Keras `model.fit(class_weight=...)`.
            3. **Real-time Data Augmentation**:
               Random rotations, flips, zooms, and contrast shifts applied strictly to training batches.
            4. **Regularization & Generalization**:
               Batch Normalization, L2 weight penalties, and progressive Dropout (0.20 to 0.50) to prevent overfitting.
            5. **Adaptive Optimization**:
               Adam optimizer with `ReduceLROnPlateau` and `EarlyStopping` monitoring validation loss.
            """
        )

    # -------------------------------------------------------------
    # TAB 4: MEDICAL & ETHICAL DISCLAIMER
    # -------------------------------------------------------------
    with tab_disclaimer:
        st.subheader("⚖️ Comprehensive Educational and Medical Disclaimer")
        st.error(
            """
            ### IMPORTANT NOTICE FOR ALL USERS AND EVALUATORS:
            
            1. **Academic Demonstration Only**:
               This software is developed strictly as a computer science undergraduate academic project.
               It has **NOT** undergone medical device certification, clinical trials, or validation by regulatory
               bodies (such as the US FDA, CE mark, or Indian CDSCO).
               
            2. **Not a Clinical Diagnostic Tool**:
               This tool must **NEVER** be used to diagnose, treat, or manage any medical condition or skin lesion.
               Automated machine learning predictions are statistical approximations of image pixels and can produce
               false positives and false negatives.
               
            3. **Consult Qualified Professionals**:
               If you or anyone you know has a skin lesion that is changing in size, shape, or color, bleeding, itching,
               or appearing asymmetric, schedule an in-person clinical appointment with a certified dermatologist immediately.
               
            4. **Biopsy is the Gold Standard**:
               A definitive diagnosis of skin cancer (such as Melanoma or Basal Cell Carcinoma) requires histopathological
               examination of a physical tissue biopsy conducted by an accredited pathologist, not an automated computer algorithm.
            """
        )


if __name__ == "__main__":
    # If the user ran 'python app.py' directly instead of 'streamlit run app.py', provide guidance
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        if get_script_run_ctx() is None:
            print("=" * 72)
            print("  [STREAMLIT LAUNCH INSTRUCTION]")
            print("=" * 72)
            print("Streamlit applications must be launched with 'streamlit run':")
            print("  streamlit run app.py")
            print("\nPlease execute the command above in your activated terminal.")
            print("=" * 72)
            sys.exit(0)
    except Exception:
        pass
    main()
