import streamlit as st
from PIL import Image
import torch
from transformers import (
    BlipProcessor,
    BlipForConditionalGeneration,
    VisionEncoderDecoderModel,
    ViTImageProcessor,
    AutoTokenizer
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

st.set_page_config(page_title="Image Caption Generator", page_icon="🖼️", layout="centered")

st.title("Image Caption Generator")
st.write("Upload an image and generate captions using BLIP or ViT-GPT2 models.")
st.info(f"Device: {DEVICE}")

MODEL_OPTIONS = [
    "BLIP Base (fast)",
    "BLIP Large (better)",
    "ViT GPT2 (fast & fluent)"
]

caption_style = st.selectbox("Select Caption Style", ["Single-line", "Detailed"])
model_name = st.selectbox("Select Model", MODEL_OPTIONS, index=0)

uploaded_file = st.file_uploader("Upload an image (JPG/PNG)", type=["jpg", "jpeg", "png"])

if caption_style == "Single-line":
    max_length, num_beams = 25, 4
else:
    max_length, num_beams = 80, 6


@st.cache_resource
def load_blip(model_id: str):
    processor = BlipProcessor.from_pretrained(model_id)
    model = BlipForConditionalGeneration.from_pretrained(model_id).to(DEVICE)
    model.eval()
    return processor, model


@st.cache_resource
def load_vit_gpt2(model_id: str):
    feature_extractor = ViTImageProcessor.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = VisionEncoderDecoderModel.from_pretrained(model_id).to(DEVICE)
    model.eval()
    return feature_extractor, tokenizer, model


def generate_caption(image: Image.Image, model_choice: str):
    gen_kwargs = {
        "max_length": max_length,
        "num_beams": num_beams,
        "no_repeat_ngram_size": 3,
        "early_stopping": True
    }

    # BLIP Base
    if model_choice.startswith("BLIP Base"):
        processor, model = load_blip("Salesforce/blip-image-captioning-base")
        inputs = processor(images=image, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            ids = model.generate(**inputs, **gen_kwargs)
        return processor.decode(ids[0], skip_special_tokens=True)

    # BLIP Large
    if model_choice.startswith("BLIP Large"):
        processor, model = load_blip("Salesforce/blip-image-captioning-large")
        inputs = processor(images=image, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            ids = model.generate(**inputs, **gen_kwargs)
        return processor.decode(ids[0], skip_special_tokens=True)

    # ViT GPT2
    feature_extractor, tokenizer, model = load_vit_gpt2("nlpconnect/vit-gpt2-image-captioning")
    pixel_values = feature_extractor(images=image, return_tensors="pt").pixel_values.to(DEVICE)
    with torch.no_grad():
        ids = model.generate(pixel_values, **gen_kwargs)
    return tokenizer.decode(ids[0], skip_special_tokens=True).strip()


# UI
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", width="stretch")  # ✅ updated

    if st.button("Generate Caption"):
        try:
            with st.spinner("Loading selected model (first time may take 1-3 minutes)..."):
                caption = generate_caption(image, model_name)

            st.subheader("Generated Caption")
            st.success(caption)

        except RuntimeError as e:
            st.error("Model failed to load. This usually happens due to low RAM on Streamlit Cloud.")
            st.exception(e)

        except Exception as e:
            st.error("Unexpected error occurred.")
            st.exception(e)

else:
    st.warning("Please upload an image to generate a caption.")
