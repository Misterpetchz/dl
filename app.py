import tempfile
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import time
import io
import streamlit as st
import pandas as pd
import requests
from io import BytesIO
from huggingface_hub import hf_hub_download
import timm
import plotly.express as px

# Use a temporary directory for uploads
UPLOAD_FOLDER = tempfile.gettempdir()
LABELS = ['Alucan', 'HDPEM', 'PET', 'Glass']

MODEL_CONFIGS = {
    "model1": {
        "name": "Waste EfficientNet",
        "repo_id": "petcht1507/waste-classification",
        "filename": "best_model_efficientnet.pth",
        "architecture": "tf_efficientnetv2_s.in21k_ft_in1k",
        "num_classes": len(LABELS)
    },
    "model2": {
        "name": "Waste ResNet101",
        "repo_id": "petcht1507/waste-classification",
        "filename": "best_model_resnet.pth",
        "architecture": "resnet101",
        "num_classes": len(LABELS)
    }
}

@st.cache_resource
def load_model(model_key):
    config = MODEL_CONFIGS[model_key]
    model = timm.create_model(config["architecture"], pretrained=False, num_classes=len(LABELS))
    model_path = hf_hub_download(repo_id=config["repo_id"], filename=config["filename"])
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    model.eval()
    return model

def predict_image(image_data, model_key):
    model = load_model(model_key)
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    img = Image.open(io.BytesIO(image_data)).convert('RGB')
    img_tensor = transform(img).unsqueeze(0)
    
    start_time = time.time()
    with torch.no_grad():
        output = model(img_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)[0]
    end_time = time.time()
    inference_time = end_time - start_time
    
    top_probs, top_indices = torch.topk(probabilities, len(LABELS))
    results = [{
        'class': LABELS[idx.item()],
        'probability': float(prob.item()) * 100,
        'inference_time': inference_time
    } for prob, idx in zip(top_probs, top_indices)]
    
    return results

st.set_page_config(page_title="Waste Image Classifier", page_icon="🗑️", layout="wide")
st.title("Waste Image Classifier - Model Comparison")
st.markdown("Upload an image to classify the waste material using two models.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Center the uploaded image
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        st.image(uploaded_file, caption="Uploaded Image", width=400)

    if st.button("Classify with Both Models"):
        col1, col2 = st.columns(2)
        img_bytes = uploaded_file.getvalue()
        
        predictions1 = predict_image(img_bytes, "model1")
        predictions2 = predict_image(img_bytes, "model2")
        
        with col1:
            st.subheader(f"Model 1: {MODEL_CONFIGS['model1']['name']}")
            with st.expander("ℹ️ Hover for Model Summary", expanded=False):
                st.text("Architecture: " + MODEL_CONFIGS["model1"]["architecture"])
            st.table(pd.DataFrame(predictions1).drop(columns=['inference_time']))

        with col2:
            st.subheader(f"Model 2: {MODEL_CONFIGS['model2']['name']}")
            with st.expander("ℹ️ Hover for Model Summary", expanded=False):
                st.text("Architecture: " + MODEL_CONFIGS["model2"]["architecture"])
            st.table(pd.DataFrame(predictions2).drop(columns=['inference_time']))

        # Inference time comparison

        inference_times_df = pd.DataFrame({
            "Model": [MODEL_CONFIGS["model1"]["name"], MODEL_CONFIGS["model2"]["name"]],
            "Inference Time (s)": [
                predictions1[0]['inference_time'],
                predictions2[0]['inference_time']
            ]
        })

        # Create bar chart for inference time comparison
        fig = px.bar(
            inference_times_df, 
            x="Model", 
            y="Inference Time (s)", 
            title="Inference Time Comparison", 
            text_auto=True,
            color="Model"
        )
        st.plotly_chart(fig)
