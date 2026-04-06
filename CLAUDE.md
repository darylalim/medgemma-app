# CLAUDE.md

## Project

Streamlit app for medical image and text inference using Google's MedGemma 1.5 4B model in MLX format, running locally on Mac with Apple Silicon.

## Commands

```bash
uv sync                           # Install dependencies
uv run streamlit run streamlit_app.py  # Run the app
uv run ruff check .               # Lint
uv run ruff format .              # Format
uv run ty check                   # Type check
uv run pytest                     # Run tests
```

## Architecture

Single-file app (`streamlit_app.py`) with the following structure:

- **Model loading** — `@st.cache_resource` loads `mlx-community/medgemma-1.5-4b-it-bf16` via `mlx_vlm.load()` + `load_config()`. Returns `(model, processor, config)`.
- **Helper functions** — Pure functions extracted for testability, called by `main()`:
  - `parse_response(response, is_thinking)` — splits `<unused94>`/`<unused95>` thinking trace from response
  - `build_messages(prompt, system_instruction, image)` — constructs chat message list for `apply_chat_template`
  - `get_generation_params(mode, is_thinking, system_instruction)` — returns `(full_instruction, max_new_tokens)`
- **Sidebar UI** — Mode toggle (Image+Text / Text Only), thinking toggle, editable system instruction.
- **Inference** — `main()` uses the helpers to build messages, format via `apply_chat_template`, call `generate()`, and parse the response.

Tests in `tests/test_streamlit_app.py` cover the helper functions without Streamlit or model mocking.

## Constraints

- **No multi-turn chat** — single Q&A per interaction
- **Package management** — uv (`pyproject.toml` + `uv.lock`); no `requirements.txt`
- **HF token** — loaded from `.env` via `python-dotenv`; required for gated model download
- **Streamlit API** — use `width="stretch"` (not deprecated `use_container_width`)
