import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import io
from huggingface_hub import hf_hub_download
import os
import time
import timm

labels = ["AluCan", "Glass", "HDPEM", "PET"]

configs = {
    "repo_id": "petcht1507/waste-classification",
    "num_classes": len(labels),
    "model1": {
        "name": "tf_efficientnetv2_s.in21k_ft_in1k",
        "file_name": "best_model_effnetv2-v2.pth",
    },
    "model2": {
        "name": "",
        "file_name": "",
    }
}

# Load the model
@st.cache_resource
def load_model():
    print("Loading model...")