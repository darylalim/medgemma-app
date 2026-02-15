# MedGemma Streamlit App Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a single-file Streamlit app that runs MedGemma 1.5 4B locally on Mac (MPS) for medical image+text and text-only inference.

**Architecture:** Single `streamlit_app.py` with cached model loading via HF `pipeline`, sidebar for mode/settings, main area for input/output. No quantization (MPS). Single Q&A interaction.

**Tech Stack:** Streamlit, PyTorch (MPS), Hugging Face Transformers pipeline, python-dotenv, Pillow

---

### Task 1: Update requirements.txt

**Files:**
- Modify: `requirements.txt`

**Step 1: Update requirements.txt with all dependencies**

Replace contents with:

```
accelerate
Pillow
python-dotenv
streamlit
torch
transformers
```

**Step 2: Verify dependencies are installed**

Run: `pip install -r requirements.txt`
Expected: All satisfied (already installed)

**Step 3: Commit**

```bash
git add requirements.txt
git commit -m "chore: update requirements with all app dependencies"
```

---

### Task 2: Build the Streamlit app — model loading

**Files:**
- Create: `streamlit_app.py`

**Step 1: Write model loading section**

Write `streamlit_app.py` with imports, env loading, and cached model loader:

```python
import os

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
        model_kwargs={
            "dtype": torch.bfloat16,
            "device_map": "auto",
        },
    )
    return pipe
```

**Step 2: Verify it loads without errors**

Run: `streamlit run streamlit_app.py`
Expected: App starts, no import errors. Page is blank (no UI yet). Stop with Ctrl+C.

**Step 3: Commit**

```bash
git add streamlit_app.py
git commit -m "feat: add model loading with cached HF pipeline"
```

---

### Task 3: Build the Streamlit app — sidebar UI

**Files:**
- Modify: `streamlit_app.py`

**Step 1: Add sidebar and page config**

Add page config at the top (after imports) and sidebar controls in a `main()` function:

```python
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

    # Input section
    prompt = st.text_input("Enter your question", placeholder="e.g. Describe this X-ray")

    uploaded_image = None
    if mode == "Image + Text":
        uploaded_file = st.file_uploader(
            "Upload a medical image", type=["png", "jpg", "jpeg", "webp"]
        )
        if uploaded_file is not None:
            uploaded_image = Image.open(uploaded_file)
            st.image(uploaded_image, caption="Uploaded image", use_container_width=True)


if __name__ == "__main__":
    main()
```

**Step 2: Verify the sidebar renders**

Run: `streamlit run streamlit_app.py`
Expected: Page shows title, sidebar with mode radio, thinking toggle, system instruction. Main area has text input and image uploader (when in Image+Text mode). Switching to "Text Only" hides the uploader.

**Step 3: Commit**

```bash
git add streamlit_app.py
git commit -m "feat: add sidebar UI with mode toggle and settings"
```

---

### Task 4: Build the Streamlit app — inference and output

**Files:**
- Modify: `streamlit_app.py`

**Step 1: Add inference logic and response display**

Add the generate button, message building, inference call, and response parsing inside `main()`, after the input section:

```python
    generate = st.button("Generate", type="primary", disabled=not prompt)

    if generate and prompt:
        # Build system instruction
        if is_thinking:
            full_instruction = f"SYSTEM INSTRUCTION: think silently if needed. {system_instruction}"
            max_new_tokens = 1300
        else:
            full_instruction = system_instruction
            max_new_tokens = 300 if mode == "Image + Text" else 500

        # Build messages
        user_content = [{"type": "text", "text": prompt}]
        if mode == "Image + Text" and uploaded_image is not None:
            user_content.append({"type": "image", "image": uploaded_image})

        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": full_instruction}],
            },
            {
                "role": "user",
                "content": user_content,
            },
        ]

        # Run inference
        with st.spinner("Generating response..."):
            output = pipe(text=messages, max_new_tokens=max_new_tokens, do_sample=False)
            response = output[0]["generated_text"][-1]["content"]

        # Parse and display response
        if is_thinking and "<unused95>" in response:
            thought, response = response.split("<unused95>", 1)
            thought = thought.replace("<unused94>thought\n", "")
            with st.expander("Thinking trace"):
                st.markdown(thought)

        st.markdown("### Response")
        st.markdown(response)

    elif generate and mode == "Image + Text" and uploaded_image is None:
        st.warning("Please upload an image for Image + Text mode.")
```

**Step 2: Test text-only mode end-to-end**

Run: `streamlit run streamlit_app.py`
1. Select "Text Only" in sidebar
2. Type "How do you differentiate bacterial from viral pneumonia?"
3. Click Generate
Expected: Spinner shows while model runs, then response appears below.

**Step 3: Test image+text mode end-to-end**

1. Select "Image + Text" in sidebar
2. Upload a chest X-ray image
3. Type "Describe this X-ray"
4. Click Generate
Expected: Image displays, spinner shows, response appears.

**Step 4: Commit**

```bash
git add streamlit_app.py
git commit -m "feat: add inference and response display"
```

---

### Task 5: Add .gitignore and clean up

**Files:**
- Create: `.gitignore`
- Modify: `requirements.txt` (if needed)

**Step 1: Create .gitignore**

```
.env
.venv/
__pycache__/
*.pyc
.ipynb_checkpoints/
```

**Step 2: Verify no secrets are tracked**

Run: `git status`
Expected: `.env` and `.venv/` are not listed as tracked files.

**Step 3: Commit**

```bash
git add .gitignore
git commit -m "chore: add .gitignore"
```
