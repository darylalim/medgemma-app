# MedGemma Streamlit App Design

## Summary

Single-file Streamlit app (`streamlit_app.py`) that wraps Google's MedGemma 1.5 4B model for medical image and text inference, running locally on Mac with MPS.

## Constraints

- **Runtime**: Mac with Apple Silicon (MPS backend)
- **Model**: `google/medgemma-1.5-4b-it` (hardcoded)
- **No quantization**: bitsandbytes does not support MPS
- **Single Q&A**: No multi-turn chat history

## Architecture

Single file with three logical sections:

1. **Model loading** — `@st.cache_resource`, loads via `pipeline("image-text-to-text")`, `device_map="auto"`, `torch.bfloat16`
2. **UI** — Sidebar for mode/settings, main area for input/output
3. **Inference** — Builds chat messages, calls `pipe()`, parses and displays response

## UI Layout

### Sidebar
- Mode toggle: "Image + Text" / "Text Only"
- Thinking toggle (on/off)
- System instruction text area (pre-filled per mode: "You are an expert radiologist" for image mode, "You are a helpful medical assistant" for text mode)

### Main Area
- Text input for the user's question
- Image uploader (visible only in Image+Text mode)
- "Generate" button
- Results area with response (and thinking trace if enabled)

## Data Flow

```
User input (text + optional image)
  -> Build messages list (system + user content)
  -> pipe(messages, max_new_tokens, do_sample=False)
  -> Parse response (split thinking trace if enabled)
  -> Display in Streamlit
```

## Key Decisions

- **`pipeline` API** over direct model loading — simpler, less code
- **HF token** from `.env` via `python-dotenv`
- **`@st.cache_resource`** for model persistence across reruns
- **`st.spinner`** during inference for user feedback
- **`do_sample=False`** for deterministic outputs (greedy decoding)
- **Thinking mode**: system instruction prefix `"SYSTEM INSTRUCTION: think silently if needed."`, response parsed by splitting on `<unused95>` token

## Dependencies

- streamlit
- torch
- transformers
- accelerate
- Pillow
- python-dotenv
