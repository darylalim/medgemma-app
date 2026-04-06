import streamlit as st
import torch
from dotenv import load_dotenv
from PIL import Image
from transformers import pipeline

load_dotenv()

MODEL_ID = "google/medgemma-1.5-4b-it"


@st.cache_resource
def load_model():
    pipe = pipeline(
        "image-text-to-text",
        model=MODEL_ID,
        model_kwargs={"dtype": torch.bfloat16, "device_map": "auto"},
    )
    pipe.model.generation_config.pad_token_id = pipe.tokenizer.eos_token_id  # ty: ignore[unresolved-attribute]
    pipe.model.generation_config.max_length = None
    pipe.model.generation_config.do_sample = False
    return pipe


def parse_response(response: str, is_thinking: bool) -> tuple[str | None, str]:
    if is_thinking and "<unused95>" in response:
        thought, answer = response.split("<unused95>", 1)
        thought = thought.removeprefix("<unused94>thought\n")
        return thought, answer
    return None, response


def build_messages(
    prompt: str, system_instruction: str, image: Image.Image | None
) -> list:
    user_content: list[dict] = [{"type": "text", "text": prompt}]
    if image is not None:
        user_content.append({"type": "image"})
    return [
        {"role": "system", "content": system_instruction},
        {"role": "user", "content": user_content},
    ]


def get_generation_params(
    mode: str, is_thinking: bool, system_instruction: str
) -> tuple[str, int]:
    if is_thinking:
        return (
            f"SYSTEM INSTRUCTION: think silently if needed. {system_instruction}",
            1300,
        )
    max_new_tokens = 300 if mode == "Image + Text" else 500
    return system_instruction, max_new_tokens


st.set_page_config(page_title="MedGemma", layout="wide")


def main():
    st.title("MedGemma")
    st.caption("Medical image and text analysis powered by MedGemma 1.5 4B")

    with st.sidebar:
        st.header("Settings")
        mode = st.radio("Mode", ["Image + Text", "Text Only"])
        is_thinking = st.toggle("Enable thinking", value=False)
        default_instruction = (
            "You are an expert radiologist."
            if mode == "Image + Text"
            else "You are a helpful medical assistant."
        )
        system_instruction = st.text_area(
            "System instruction", value=default_instruction, height=100
        )

    with st.spinner("Loading model..."):
        pipe = load_model()

    prompt = st.text_input(
        "Enter your question", placeholder="e.g. Describe this X-ray"
    )

    uploaded_image = None
    if mode == "Image + Text":
        uploaded_file = st.file_uploader(
            "Upload a medical image", type=["png", "jpg", "jpeg", "webp"]
        )
        if uploaded_file is not None:
            try:
                uploaded_image = Image.open(uploaded_file)
                st.image(uploaded_image, caption="Uploaded image", width="stretch")
            except Exception:
                st.error("Failed to load image. Please upload a valid image file.")

    generate = st.button("Generate", type="primary", disabled=not prompt)

    if generate and prompt:
        full_instruction, max_new_tokens = get_generation_params(
            mode, is_thinking, system_instruction
        )
        messages = build_messages(
            prompt,
            full_instruction,
            uploaded_image if mode == "Image + Text" else None,
        )

        with st.spinner("Generating response..."):
            try:
                output = pipe(text=messages, max_new_tokens=max_new_tokens)
                response = output[0]["generated_text"][-1]["content"]
            except Exception as e:
                st.error(f"Inference failed: {e}")
                return

        thought, response = parse_response(response, is_thinking)
        if thought is not None:
            with st.expander("Thinking trace"):
                st.markdown(thought)

        st.markdown("### Response")
        st.markdown(response)

    elif generate and mode == "Image + Text" and uploaded_image is None:
        st.warning("Please upload an image for Image + Text mode.")


if __name__ == "__main__":
    main()
