import streamlit as st
from PIL import Image
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import torch
from transformers import (
    BlipForConditionalGeneration,
    BlipProcessor,
    VisionEncoderDecoderModel,
    ViTImageProcessor,
    AutoTokenizer
)

# Device setup
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load models (cached so it won't reload every time)
@st.cache_resource
def load_models():
    models = {}

    # BLIP Base
    blip_base_id = "Salesforce/blip-image-captioning-base"
    models["BLIP Base"] = {
        "type": "blip",
        "processor": BlipProcessor.from_pretrained(blip_base_id),
        "model": BlipForConditionalGeneration.from_pretrained(blip_base_id).to(DEVICE).eval()
    }

    # BLIP Large
    blip_large_id = "Salesforce/blip-image-captioning-large"
    models["BLIP Large"] = {
        "type": "blip",
        "processor": BlipProcessor.from_pretrained(blip_large_id),
        "model": BlipForConditionalGeneration.from_pretrained(blip_large_id).to(DEVICE).eval()
    }

    # ViT GPT2
    vit_gpt2_id = "nlpconnect/vit-gpt2-image-captioning"
    models["ViT GPT2"] = {
        "type": "vitgpt2",
        "feature_extractor": ViTImageProcessor.from_pretrained(vit_gpt2_id),
        "tokenizer": AutoTokenizer.from_pretrained(vit_gpt2_id),
        "model": VisionEncoderDecoderModel.from_pretrained(vit_gpt2_id).to(DEVICE).eval()
    }

    return models


def generate_caption_blip(image, processor, model, max_length, num_beams):
    inputs = processor(images=image, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_length=max_length,
            num_beams=num_beams,
            no_repeat_ngram_size=3,
            early_stopping=True
        )
    caption = processor.decode(output_ids[0], skip_special_tokens=True)
    return caption


def generate_caption_vitgpt2(image, feature_extractor, tokenizer, model, max_length, num_beams):
    pixel_values = feature_extractor(images=image, return_tensors="pt").pixel_values.to(DEVICE)
    with torch.no_grad():
        output_ids = model.generate(
            pixel_values,
            max_length=max_length,
            num_beams=num_beams,
            no_repeat_ngram_size=3,
            early_stopping=True
        )
    caption = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return caption


# UI
st.set_page_config(page_title="Image Caption Generator", page_icon="🖼️", layout="centered")

st.title("Image Caption Generator")
st.write("Upload an image and generate captions using BLIP and ViT GPT2 models.")

st.info(f"Device: {DEVICE}")

models = load_models()

uploaded_file = st.file_uploader("Upload an image (JPG/PNG)", type=["jpg", "jpeg", "png"])

model_name = st.selectbox("Select Model", list(models.keys()))

caption_style = st.selectbox("Select Caption Style", ["Single-line", "Detailed"])

if caption_style == "Single-line":
    max_length, num_beams = 25, 4
else:
    max_length, num_beams = 80, 6

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("Generate Caption"):
        with st.spinner("Generating caption..."):
            selected = models[model_name]

            if selected["type"] == "blip":
                caption = generate_caption_blip(
                    image,
                    selected["processor"],
                    selected["model"],
                    max_length,
                    num_beams
                )
            else:
                caption = generate_caption_vitgpt2(
                    image,
                    selected["feature_extractor"],
                    selected["tokenizer"],
                    selected["model"],
                    max_length,
                    num_beams
                )

        st.subheader("Generated Caption")
        st.success(caption)
