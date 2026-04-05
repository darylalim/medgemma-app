# MedGemma App

Streamlit web application for generating responses from medical text and images using Google's [MedGemma](https://huggingface.co/google/medgemma-1.5-4b-it) model.

## Setup

Requires [uv](https://docs.astral.sh/uv/) and a Hugging Face token with access to `google/medgemma-1.5-4b-it`.

```bash
uv sync
```

Create a `.env` file:

```
HF_TOKEN=your_token_here
```

## Usage

```bash
uv run streamlit run streamlit_app.py
```

## Testing

```bash
uv run pytest
```
