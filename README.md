# MedGemma App

Streamlit web application for generating responses from medical text and images using Google's [MedGemma](https://huggingface.co/google/medgemma-1.5-4b-it) model.

## Setup

```bash
pip install -r requirements.txt
```

Requires a Hugging Face token with access to `google/medgemma-1.5-4b-it`. Create a `.env` file:

```
HF_TOKEN=your_token_here
```

## Usage

```bash
streamlit run streamlit_app.py
```

## Testing

```bash
pytest
```
