import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
from transformers import BartModel, BartTokenizer
from torchvision import models
import pandas as pd

# ===================== 🌟 Custom CSS for Beautiful UI =====================
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;500;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif;
        background-color: #f9fafc;
    }

    h1, h2, h3 {
        color: #1f2937;
    }

    .stButton>button {
        background-color: #2563eb;
        color: white;
        border-radius: 0.5rem;
        padding: 0.6rem 1.2rem;
        font-weight: 600;
        transition: background-color 0.3s ease;
    }

    .stButton>button:hover {
        background-color: #1d4ed8;
    }

    .stTextArea textarea, .stFileUploader {
        background-color: #ffffff;
        border: 1px solid #d1d5db;
        border-radius: 0.5rem;
        padding: 10px;
    }

    .prediction-box {
        background-color: #ecfdf5;
        color: #065f46;
        padding: 1.2rem;
        border-radius: 0.75rem;
        border: 1px solid #10b981;
        font-size: 1.1rem;
        font-weight: 500;
    }

    .block-container {
        padding: 2rem 3rem;
    }
    </style>
""", unsafe_allow_html=True)

# ===================== 🧠 Model & Helper Functions =====================
class MultiModalClassifier(nn.Module):
    def __init__(self, text_model, image_model, text_feat_dim, image_feat_dim, hidden_dim, num_classes):
        super(MultiModalClassifier, self).__init__()
        self.text_model = text_model
        self.image_model = image_model
        self.text_fc = nn.Linear(text_feat_dim, hidden_dim)
        self.image_fc = nn.Linear(image_feat_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, text_input=None, image_input=None):
        features = None
        if text_input is not None:
            text_input_filtered = {k: v for k, v in text_input.items() if k != "labels"}
            text_outputs = self.text_model(**text_input_filtered)
            pooled_text = text_outputs.last_hidden_state.mean(dim=1)
            text_features = self.text_fc(pooled_text)
            features = text_features if features is None else features + text_features
        if image_input is not None:
            image_features = self.image_model(image_input)
            image_features = self.image_fc(image_features)
            features = image_features if features is None else features + image_features
        if (text_input is not None) and (image_input is not None):
            features = features / 2
        logits = self.classifier(features)
        return logits

@st.cache_resource
def load_model():
    model_name = "facebook/bart-base"
    tokenizer = BartTokenizer.from_pretrained(model_name)
    image_model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    image_model.fc = nn.Identity()
    text_df = pd.read_csv("dataset.csv")
    label_list = text_df['labels'].unique().tolist()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiModalClassifier(
        text_model=BartModel.from_pretrained(model_name),
        image_model=image_model,
        text_feat_dim=768,
        image_feat_dim=512,
        hidden_dim=512,
        num_classes=len(label_list)
    )
    model.load_state_dict(torch.load("multimodal_model.pth", map_location=device))
    model.to(device)
    model.eval()
    return model, tokenizer, label_list, device

model, tokenizer, label_list, device = load_model()
label2id = {label: idx for idx, label in enumerate(label_list)}
id2label = {idx: label for label, idx in label2id.items()}
image_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Inference functions
def inference_text(model, tokenizer, text, device, max_length=128):
    encoding = tokenizer(text, padding="max_length", truncation=True, max_length=max_length, return_tensors="pt")
    for key in encoding:
        encoding[key] = encoding[key].to(device)
    with torch.no_grad():
        logits = model(text_input=encoding, image_input=None)
    pred_id = torch.argmax(logits, dim=1).item()
    return id2label[pred_id]

def inference_image(model, image, transform, device):
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(text_input=None, image_input=image)
    pred_id = torch.argmax(logits, dim=1).item()
    return id2label[pred_id]

def inference_both(model, tokenizer, text, image, transform, device, max_length=128):
    encoding = tokenizer(text, padding="max_length", truncation=True, max_length=max_length, return_tensors="pt")
    for key in encoding:
        encoding[key] = encoding[key].to(device)
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(text_input=encoding, image_input=image)
    pred_id = torch.argmax(logits, dim=1).item()
    return id2label[pred_id]

# ===================== 🚀 App Routing =====================
st.sidebar.title("🔗 Navigation")
page = st.sidebar.selectbox("Go to", ["🏠 Home", "🩺 Classifier"])

# ===================== 🏠 Home Page =====================
if page == "🏠 Home":
    st.title("🏠 Welcome to the MultiModal Disease Classifier")
    st.markdown("""
        This application uses a **multi-modal deep learning model** that accepts:
        - 📝 **Text (Symptoms)**
        - 🖼️ **Medical Images**
        - 📊 **Both together**

        The model analyzes your input and predicts the **most likely disease** or **medical condition**.

        ---
        Click on **🩺 Classifier** in the sidebar to start diagnosing.
    """)
    st.image("https://cdn-icons-png.flaticon.com/512/2920/2920277.png", width=300)

# ===================== 🤖 Classifier Page =====================
elif page == "🩺 Classifier":
    st.title("🩺 MultiModal Disease Classifier")
    st.markdown("Diagnose with AI — Provide **symptoms**, **medical images**, or **both** to predict possible conditions.")

    option = st.sidebar.radio("Choose Input Method", ("Text Only", "Image Only", "Text and Image"))
    st.sidebar.markdown(f"📱 **Device in use:** `{device.type.upper()}`")

    st.subheader("🔎 Input Section")
    text_input = None
    image_input = None

    if option in ["Text Only", "Text and Image"]:
        text_input = st.text_area("📝 Enter Symptoms", placeholder="E.g. Headache, Chest pain, Fatigue", height=100)

    if option in ["Image Only", "Text and Image"]:
        uploaded_file = st.file_uploader("🖼️ Upload Medical Image", type=["jpg", "jpeg", "png"])
        if uploaded_file:
            image_input = Image.open(uploaded_file).convert("RGB")
            st.image(image_input, caption="📷 Uploaded Image Preview", width=300)

    if st.button("🔍 Predict", use_container_width=True):
        with st.spinner("Analyzing..."):
            if option == "Text Only" and text_input:
                prediction = inference_text(model, tokenizer, text_input, device)
                st.markdown(f"<div class='prediction-box'><strong>Prediction (Text):</strong> {prediction}</div>", unsafe_allow_html=True)
            elif option == "Image Only" and image_input:
                prediction = inference_image(model, image_input, image_transforms, device)
                st.markdown(f"<div class='prediction-box'><strong>Prediction (Image):</strong> {prediction}</div>", unsafe_allow_html=True)
            elif option == "Text and Image" and text_input and image_input:
                prediction = inference_both(model, tokenizer, text_input, image_input, image_transforms, device)
                st.markdown(f"<div class='prediction-box'><strong>Prediction (Combined):</strong> {prediction}</div>", unsafe_allow_html=True)
            else:
                st.error("🚨 Please provide the required inputs based on selected method.")

    st.markdown("---")
    st.caption("🧠 Built with ❤️ by xAI | Ensure `dataset.csv` and `multimodal_model.pth` are available.")
