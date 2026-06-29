import json
import os

import streamlit as st
st.set_page_config(page_title="CaptionLens", layout="wide")

from PIL import Image
import torch
import yaml
import torchvision.transforms as transforms

from models.image_captioning import ImageCaptioningModel
from utils.decoding import generate_caption as decode_image
from utils.tokenizer import Tokenizer


st.markdown(
    """
    <style>
        :root {
            --deep-blue: #0f2f57;
            --blue: #2563eb;
            --soft-blue: #eaf3ff;
            --border: #d8e4f2;
            --text: #172033;
            --muted: #64748b;
            --card: rgba(255, 255, 255, 0.92);
        }

        .stApp {
            background:
                radial-gradient(circle at top left, rgba(80, 145, 255, 0.18), transparent 32rem),
                linear-gradient(135deg, #f6f9ff 0%, #eef5ff 45%, #f8fbff 100%);
            color: var(--text);
        }

        .block-container {
            max-width: 1180px;
            padding-top: 2.2rem;
            padding-bottom: 3rem;
        }

        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0f2f57 0%, #173f73 100%);
        }

        section[data-testid="stSidebar"] * {
            color: #f8fbff !important;
        }

        .hero {
            text-align: center;
            padding: 2.1rem 1.5rem 1.25rem;
        }

        .hero-icon {
            width: 3.2rem;
            height: 3.2rem;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            border-radius: 1rem;
            background: linear-gradient(135deg, #dbeafe, #ffffff);
            border: 1px solid rgba(37, 99, 235, 0.16);
            box-shadow: 0 12px 32px rgba(37, 99, 235, 0.14);
            font-size: 1.75rem;
            margin-bottom: 0.8rem;
        }

        .hero-title {
            margin: 0;
            font-size: clamp(2.6rem, 6vw, 4.4rem);
            font-weight: 850;
            line-height: 1.02;
            letter-spacing: 0;
            background: linear-gradient(90deg, #0f2f57, #2563eb 52%, #14b8a6);
            -webkit-background-clip: text;
            background-clip: text;
            color: transparent;
        }

        .hero-subtitle {
            margin: 0.8rem auto 0;
            max-width: 820px;
            color: #233852;
            font-size: 1.16rem;
            font-weight: 650;
            line-height: 1.45;
        }

        .hero-cn {
            margin: 0.45rem auto 0;
            color: #46617f;
            font-size: 1rem;
            line-height: 1.5;
        }

        .hero-desc {
            margin: 0.95rem auto 0;
            max-width: 780px;
            color: var(--muted);
            font-size: 1rem;
            line-height: 1.6;
        }

        .badge-row {
            display: flex;
            justify-content: center;
            flex-wrap: wrap;
            gap: 0.55rem;
            margin-top: 1.25rem;
        }

        .badge {
            padding: 0.42rem 0.7rem;
            border-radius: 999px;
            border: 1px solid rgba(37, 99, 235, 0.18);
            background: rgba(255, 255, 255, 0.74);
            color: #1e4777;
            font-size: 0.82rem;
            font-weight: 700;
            box-shadow: 0 8px 20px rgba(15, 47, 87, 0.06);
        }

        .section-title {
            margin: 1.8rem 0 0.75rem;
            color: var(--deep-blue);
            font-size: 1.35rem;
            font-weight: 800;
            letter-spacing: 0;
        }

        .panel-title {
            margin: 0 0 0.75rem;
            color: var(--deep-blue);
            font-size: 1.08rem;
            font-weight: 800;
        }

        .soft-card {
            padding: 1.2rem;
            border: 1px solid var(--border);
            border-radius: 1.25rem;
            background: var(--card);
            box-shadow: 0 18px 42px rgba(15, 47, 87, 0.10);
        }

        .caption-box {
            margin-top: 0.8rem;
            padding: 1.05rem 1.1rem;
            border-radius: 1rem;
            border: 1px solid #bfdbfe;
            background: linear-gradient(135deg, #eff6ff, #ffffff);
            color: #10294c;
            font-size: 1.28rem;
            font-weight: 750;
            line-height: 1.55;
            box-shadow: inset 0 0 0 1px rgba(255, 255, 255, 0.72);
        }

        .hint-card {
            padding: 1.1rem;
            border-radius: 1rem;
            border: 1px dashed #adc6e7;
            background: rgba(239, 246, 255, 0.74);
            color: #46617f;
            font-size: 0.98rem;
            line-height: 1.55;
        }

        .flow-card {
            min-height: 168px;
            padding: 1.1rem 1rem;
            border-radius: 1.05rem;
            border: 1px solid var(--border);
            background: rgba(255, 255, 255, 0.82);
            box-shadow: 0 14px 34px rgba(15, 47, 87, 0.08);
        }

        .flow-icon {
            width: 2.35rem;
            height: 2.35rem;
            display: flex;
            align-items: center;
            justify-content: center;
            border-radius: 0.8rem;
            background: #eaf3ff;
            color: #1d4ed8;
            font-size: 1.25rem;
            margin-bottom: 0.8rem;
        }

        .flow-title {
            color: var(--deep-blue);
            font-weight: 820;
            font-size: 1rem;
            margin-bottom: 0.35rem;
        }

        .flow-text {
            color: var(--muted);
            font-size: 0.92rem;
            line-height: 1.5;
        }

        div[data-testid="stFileUploader"] {
            padding: 0.75rem;
            border-radius: 1rem;
            background: #f7fbff;
            border: 1px solid #ddeafe;
        }

        div[data-testid="stImage"] img {
            border-radius: 1rem;
            border: 1px solid var(--border);
            box-shadow: 0 14px 36px rgba(15, 47, 87, 0.12);
        }

        div[data-testid="stSpinner"] {
            color: #1d4ed8;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# Load config
with open("config/config.yaml", 'r') as f:
    config = yaml.safe_load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vocab_path = config['processed_data']['vocab_path']
checkpoint_path = os.path.join(config['train']['save_dir'], "best_model.pth")


tokenizer = None
model = None

if not os.path.exists(vocab_path):
    st.error("Vocabulary file not found. Please run build_vocab.py first.")
else:
    try:
        # Load tokenizer
        tokenizer = Tokenizer()
        tokenizer.load_vocab(vocab_path)
    except Exception as exc:
        st.error("Failed to load vocabulary file. Please rebuild it with build_vocab.py.")
        st.caption(str(exc))
        tokenizer = None


# Load model
@st.cache_resource
def load_model():
    model = ImageCaptioningModel(
        vocab_size=len(tokenizer.word2idx),
        embed_size=config['model']['embed_size'],
        decoder_dim=config['model']['decoder_dim'],
        attention_dim=config['model']['attention_dim'],
        dropout=config['model']['dropout'],
        max_len=config['model']['max_len'],
        use_semantic_slots=config['model'].get('use_semantic_slots', False),
        num_semantic_slots=config['model'].get('num_semantic_slots', 16),
        semantic_slot_layers=config['model'].get('semantic_slot_layers', 2),
        semantic_slot_heads=config['model'].get('semantic_slot_heads', 8),
        include_visual_tokens=config['model'].get('include_visual_tokens', True),
    )
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    return model


if tokenizer is not None:
    if not os.path.exists(checkpoint_path):
        st.error("Model checkpoint not found. Please train the model first.")
    else:
        try:
            model = load_model()
        except Exception as exc:
            st.error("Failed to load model checkpoint. Please check whether the checkpoint matches the vocabulary and model config.")
            st.caption(str(exc))
            model = None


with st.sidebar:
    st.markdown("## CaptionLens")
    st.markdown("**Project:** CaptionLens")
    st.markdown("**Model:** ResNet50 + Semantic Slot Adapter + Transformer Decoder")
    st.markdown("**Mechanism:** Self-Attention + Cross-Attention")
    st.markdown("**Dataset:** MSCOCO 2017")
    st.markdown("**Decoding:** Beam Search")
    st.markdown(f"**Device:** `{device.type}`")
    evaluation_path = config.get("evaluation", {}).get("output_path")
    if evaluation_path and os.path.exists(evaluation_path):
        try:
            with open(evaluation_path, "r", encoding="utf-8") as file:
                evaluation = json.load(file)
            beam_metrics = evaluation.get("metrics", {}).get("beam", {})
            if beam_metrics:
                st.markdown("---")
                st.markdown("### Validation Metrics")
                st.markdown(
                    f"**BLEU-4:** `{beam_metrics['BLEU-4']:.4f}`"
                )
                st.markdown(
                    f"**CIDEr:** `{beam_metrics['CIDEr']:.4f}`"
                )
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            pass


# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])


def generate_caption(image):
    image_tensor = transform(image).unsqueeze(0).to(device)
    caption = decode_image(
        model,
        image_tensor,
        tokenizer,
        strategy="beam",
        **config["decoding"],
    )
    return caption or "Unable to generate a reliable caption."


st.markdown(
    """
    <div class="hero">
        <div class="hero-icon">📷</div>
        <h1 class="hero-title">CaptionLens</h1>
        <div class="hero-subtitle">Image Semantic Captioning with Self-Attention and Cross-Attention</div>
        <div class="hero-cn">融合自注意力与交叉注意力机制的图像语义描述生成系统</div>
        <div class="hero-desc">
            Upload an image and generate a natural language description using Transformer attention.
        </div>
        <div class="badge-row">
            <span class="badge">Transformer</span>
            <span class="badge">Self-Attention</span>
            <span class="badge">Cross-Attention</span>
            <span class="badge">MSCOCO</span>
            <span class="badge">Streamlit</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


left_col, right_col = st.columns([1.05, 0.95], gap="large")

with left_col:
    st.markdown('<div class="section-title">Upload Image</div>', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">Input Image</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Upload Image",
        type=["jpg", "png", "jpeg"],
        label_visibility="collapsed"
    )

    image = None
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

with right_col:
    st.markdown('<div class="section-title">Generated Caption</div>', unsafe_allow_html=True)

    if uploaded_file and image is not None:
        if model is None or tokenizer is None:
            st.markdown(
                '<div class="hint-card">Model resources are not ready. Please resolve the message above and rerun the app.</div>',
                unsafe_allow_html=True,
            )
        else:
            with st.spinner("Generating caption with Transformer attention..."):
                caption = generate_caption(image)

            st.markdown(
                f"""
                <div class="panel-title">Generated Caption</div>
                <div class="caption-box">{caption}</div>
                """,
                unsafe_allow_html=True,
            )
    else:
        st.markdown(
            '<div class="hint-card">Please upload an image to generate a caption.</div>',
            unsafe_allow_html=True,
        )


st.markdown('<div class="section-title">How it works</div>', unsafe_allow_html=True)
flow_cols = st.columns(4, gap="medium")
flow_items = [
    ("🧩", "CNN Encoder", "Extracts regional visual features from the input image."),
    ("🔲", "Visual Tokens", "Converts the image into 14 × 14 visual region tokens."),
    ("🔁", "Self-Attention", "Models relationships among generated words."),
    ("🎯", "Cross-Attention", "Aligns text tokens with image region features."),
]

for col, (icon, title, text) in zip(flow_cols, flow_items):
    with col:
        st.markdown(
            f"""
            <div class="flow-card">
                <div class="flow-icon">{icon}</div>
                <div class="flow-title">{title}</div>
                <div class="flow-text">{text}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
