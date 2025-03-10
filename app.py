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
LABELS = ['Alucan', 'Glass', 'HDPEM', 'PET']

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

@st.cache_resource
def get_model_summary(model_key):
    """Generate summary statistics for a model including layers and parameters"""
    model = load_model(model_key)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Get basic layer information
    layer_types = {}
    for name, module in model.named_modules():
        layer_type = module.__class__.__name__
        if layer_type in layer_types:
            layer_types[layer_type] += 1
        else:
            layer_types[layer_type] = 1
    
    # Remove the 'Sequential' and 'Module' entries as they're not specific layers
    if 'Sequential' in layer_types:
        del layer_types['Sequential']
    if 'Module' in layer_types:
        del layer_types['Module']
    
    summary = {
        "Architecture": MODEL_CONFIGS[model_key]["architecture"],
        "Total Parameters": f"{total_params:,}",
        "Trainable Parameters": f"{trainable_params:,}",
        "Layer Types": layer_types
    }
    
    return summary

def create_hover_box(model_key):
    summary = get_model_summary(model_key)
    
    layer_types_html = "".join([f"<li>{layer_type}: {count}</li>" for layer_type, count in summary['Layer Types'].items()])
    
    html = f"""
    <div class="tooltip">ℹ️ Model Details
      <div class="tooltiptext">
        <b>Architecture:</b> {summary['Architecture']}<br>
        <b>Total Parameters:</b> {summary['Total Parameters']}<br>
        <b>Trainable Parameters:</b> {summary['Trainable Parameters']}<br>
        <b>Layer Types:</b>
        <ul>{layer_types_html}</ul>
      </div>
    </div>
    
    <style>
    .tooltip {{
      position: relative;
      display: inline-block;
      border-bottom: 1px dotted black;
      cursor: help;
    }}
    
    .tooltip .tooltiptext {{
      visibility: hidden;
      width: 300px;
      background-color: #f0f2f6;
      color: black;
      text-align: left;
      border-radius: 6px;
      padding: 10px;
      position: absolute;
      z-index: 1;
      top: 125%;
      left: 0;
      transform: translateX(0);
      opacity: 0;
      transition: opacity 0.3s;
      box-shadow: 0px 0px 8px rgba(0,0,0,0.2);
    }}
    
    .tooltip:hover .tooltiptext {{
      visibility: visible;
      opacity: 1;
    }}
    </style>
    """
    return html


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
st.markdown("<h1 style='text-align: center;'>Waste Image Classifier - Model Comparison</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Upload an image to classify the waste material using two models.</p>", unsafe_allow_html=True)

col1, col2, col3 = st.columns([1, 10, 1])
with col2:
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Center the uploaded image with more weight to the central column
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
    
    # Center the button
    st.markdown(
        """
        <style>
        div.stButton > button {
            display: block;
            margin: 0 auto 0;
            min-width: 200px;
        }
        /* Center subheaders */
        .css-10trblm {
            text-align: center;
        }
        </style>
        """, 
        unsafe_allow_html=True
    )
    classify_button = st.button("Classify with Both Models")
    
    if classify_button:
        col1, col2 = st.columns(2)
        img_bytes = uploaded_file.getvalue()
        
        predictions1 = predict_image(img_bytes, "model1")
        predictions2 = predict_image(img_bytes, "model2")
        
        with col1:
            # Using markdown for centered subheader instead of st.subheader
            st.markdown(f"<h3 style='text-align: center;'>Model 1: {MODEL_CONFIGS['model1']['name']}</h3>", unsafe_allow_html=True)
            st.markdown(create_hover_box("model1"), unsafe_allow_html=True)
            st.table(pd.DataFrame(predictions1).drop(columns=['inference_time']))

        with col2:
            # Using markdown for centered subheader instead of st.subheader
            st.markdown(f"<h3 style='text-align: center;'>Model 2: {MODEL_CONFIGS['model2']['name']}</h3>", unsafe_allow_html=True)
            st.markdown(create_hover_box("model2"), unsafe_allow_html=True)
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
        st.markdown('<div style="height: 50px;"></div>', unsafe_allow_html=True)
        st.markdown("<h3 style='text-align: center;'>Inference Time Comparison</h3>", unsafe_allow_html=True)
        fig = px.bar(
            inference_times_df, 
            x="Model", 
            y="Inference Time (s)", 
            text_auto=True,
            color="Model"
        )
        st.plotly_chart(fig)