"""
RealVision Web Application
Simple and clean UI for AI-Generated Image Detection
"""

import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
from realvision_model import create_model

# Page configuration
st.set_page_config(
    page_title="RealVision",
    page_icon="🔍",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS for clean styling
st.markdown("""
<style>
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {display: none;}
    
    /* Main container styling */
    .stApp {
        background-color: #f5f7fa;
        max-width: 1400px;
        margin: 0 auto;
    }
    
    /* Remove default padding */
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 1rem !important;
        max-width: 100% !important;
    }
    
    /* Hide file uploader label */
    .stFileUploader label {
        display: none !important;
    }
    
    /* Title styling */
    .main-title {
        font-size: 2.8rem;
        font-weight: 600;
        text-align: center;
        color: #1a1a1a;
        margin-top: 0.5rem;
        margin-bottom: 0.2rem;
        letter-spacing: -0.02em;
    }
    
    .subtitle {
        font-size: 0.95rem;
        text-align: center;
        color: #6c757d;
        margin-bottom: 1.5rem;
        font-weight: 400;
    }
    
    /* Card styling */
    .upload-card, .results-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
        height: 100%;
        padding-top: 1.5rem;
    }
    
    .card-title {
        font-size: 1.3rem;
        font-weight: 600;
        color: #1a1a1a;
        margin-bottom: 0.2rem;
        margin-top: 0;
        padding-top: 0;
    }
    
    .card-subtitle {
        font-size: 0.85rem;
        color: #8e9199;
        margin-bottom: 1rem;
    }
    
    /* File uploader styling */
    .stFileUploader {
        padding: 1rem;
        border: 2px dashed #d1d5db;
        border-radius: 8px;
        background-color: #fafbfc;
        margin-bottom: 0.8rem;
    }
    
    .stFileUploader > div > div {
        text-align: center;
    }
    
    /* Image styling */
    .stImage {
        border-radius: 8px;
        margin-bottom: 0.8rem;
    }
    
    /* Top match result */
    .top-match-card {
        background: #2d3748;
        color: white;
        padding: 1.3rem;
        border-radius: 10px;
        margin-bottom: 1.2rem;
    }
    
    .top-match-label {
        font-size: 0.7rem;
        color: #a0aec0;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.5rem;
        font-weight: 600;
    }
    
    .top-match-result {
        font-size: 1.6rem;
        font-weight: 600;
        margin-bottom: 0.6rem;
        color: white;
    }
    
    .confidence-bar-container {
        height: 5px;
        background: rgba(255,255,255,0.15);
        border-radius: 3px;
        overflow: hidden;
        margin-bottom: 0.5rem;
    }
    
    .confidence-bar-fill {
        height: 100%;
        background: #48bb78;
        border-radius: 3px;
        transition: width 0.3s ease;
    }
    
    .confidence-text {
        font-size: 1.2rem;
        font-weight: 600;
        text-align: right;
        color: white;
    }
    
    /* Other predictions */
    .other-predictions-title {
        font-size: 0.95rem;
        font-weight: 600;
        color: #1a1a1a;
        margin-bottom: 0.8rem;
    }
    
    .prediction-item {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.7rem 0;
        border-bottom: 1px solid #f0f0f0;
    }
    
    .prediction-item:last-child {
        border-bottom: none;
    }
    
    .prediction-label {
        font-size: 0.9rem;
        color: #4a5568;
    }
    
    .prediction-confidence {
        font-size: 0.9rem;
        font-weight: 600;
        color: #2d3748;
    }
    
    /* Footer */
    .custom-footer {
        text-align: center;
        color: #718096;
        font-size: 0.82rem;
        margin-top: 1.5rem;
        padding: 1rem 0;
    }
    
    .custom-footer a {
        color: #3182ce;
        text-decoration: none;
    }
    
    /* Button styling */
    div[data-testid="column"] .stButton > button {
        width: 100%;
        border-radius: 6px;
        font-size: 0.9rem;
        font-weight: 500;
        padding: 0.5rem 1.2rem;
        transition: all 0.2s;
        border: 1px solid #e2e8f0;
    }
    
    /* Change Image button - white with border */
    div[data-testid="column"]:first-child .stButton > button {
        background-color: white;
        color: #4a5568;
        border: 1px solid #cbd5e0;
    }
    
    div[data-testid="column"]:first-child .stButton > button:hover {
        background-color: #f7fafc;
        border-color: #a0aec0;
    }
    
    /* Analyze Image button - black */
    div[data-testid="column"]:last-child .stButton > button {
        background-color: #1a1a1a;
        color: white;
        border: 1px solid #1a1a1a;
    }
    
    div[data-testid="column"]:last-child .stButton > button:hover {
        background-color: #2d3748;
    }
    
    /* Info message styling */
    .stInfo {
        background-color: #f0f4f8;
        border-left: 4px solid #4299e1;
        padding: 1rem;
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    """Load the trained model."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = 'checkpoints/best_model.pth'
    
    if not os.path.exists(model_path):
        st.error(f"Model not found at {model_path}. Please train the model first.")
        return None, None, None
    
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint.get('config', {})
    
    model = create_model(
        num_classes=config.get('num_classes', 2),
        dropout_rate=config.get('dropout_rate', 0.5)
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, device, checkpoint


@st.cache_data
def load_results():
    """Load evaluation results."""
    results_path = 'results/evaluation_results.json'
    if os.path.exists(results_path):
        with open(results_path, 'r') as f:
            return json.load(f)
    return None


def preprocess_image(image):
    """Preprocess image for model inference."""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)


def predict(model, image, device):
    """Make prediction on image."""
    with torch.no_grad():
        image_tensor = preprocess_image(image).to(device)
        outputs = model(image_tensor)
        probabilities = F.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
        
    return predicted.item(), confidence.item(), probabilities[0].cpu().numpy()


def main():
    # Header
    st.markdown('<div class="main-title">RealVision</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Powered by ResNet-18 • Deep Learning</div>', unsafe_allow_html=True)
    
    # Load model
    model, device, checkpoint = load_model()
    if model is None:
        st.stop()
    
    results = load_results()
    
    # Main content - Two column layout
    col1, col2 = st.columns([1, 1], gap="large")
    
    # Left column - Upload section
    with col1:
        st.markdown('<div class="card-title">Upload Image</div>', unsafe_allow_html=True)
        st.markdown('<div class="card-subtitle">Drop your image here or click to browse</div>', unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Choose an image",
            type=['png', 'jpg', 'jpeg'],
            label_visibility="collapsed"
        )
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, use_container_width=True)
            
            # Store prediction in session state
            if 'prediction_made' not in st.session_state:
                st.session_state.prediction_made = False
            
            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("🔄 Change Image", use_container_width=True):
                    st.session_state.prediction_made = False
                    st.rerun()
            with col_b:
                if st.button("⚡ Analyze Image", type="primary", use_container_width=True):
                    with st.spinner("Analyzing..."):
                        predicted_class, confidence, probabilities = predict(model, image, device)
                        st.session_state.predicted_class = predicted_class
                        st.session_state.confidence = confidence
                        st.session_state.probabilities = probabilities
                        st.session_state.prediction_made = True
                        st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Right column - Results section
    with col2:
        st.markdown('<div class="results-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Recognition Results</div>', unsafe_allow_html=True)
        st.markdown('<div class="card-subtitle">AI-powered predictions with confidence scores</div>', unsafe_allow_html=True)
        
        if uploaded_file is not None and st.session_state.get('prediction_made', False):
            predicted_class = st.session_state.predicted_class
            confidence = st.session_state.confidence
            probabilities = st.session_state.probabilities
            
            class_names = ['AI-Generated', 'Real Image']
            prediction_label = class_names[predicted_class]
            
            # Top match card
            st.markdown(f"""
            <div class="top-match-card">
                <div class="top-match-label">TOP MATCH</div>
                <div class="top-match-result">{prediction_label}</div>
                <div class="confidence-bar-container">
                    <div class="confidence-bar-fill" style="width: {confidence*100}%"></div>
                </div>
                <div class="confidence-text">{confidence*100:.2f}%</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Other predictions
            st.markdown('<div class="other-predictions-title">Other Predictions</div>', unsafe_allow_html=True)
            
            # Show both probabilities
            predictions_list = [
                ('AI-Generated', float(probabilities[0] * 100)),
                ('Real Image', float(probabilities[1] * 100))
            ]
            
            # Sort by confidence
            predictions_list.sort(key=lambda x: x[1], reverse=True)
            
            for idx, (label, prob) in enumerate(predictions_list[1:], start=2):
                st.markdown(f"""
                <div class="prediction-item">
                    <span class="prediction-label">{idx}. {label}</span>
                    <span class="prediction-confidence">{prob:.2f}%</span>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("Upload an image and click 'Analyze Image' to see results")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Footer
    st.markdown("""
    <div class="custom-footer">
        Built using PyTorch & ResNet-18<br>
        Trained on balanced dataset • Binary classification<br>
        <a href="https://github.com" target="_blank">View on GitHub</a>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    # Initialize session state
    if 'prediction_made' not in st.session_state:
        st.session_state.prediction_made = False
    
    main()


