# Remove Sidebar — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the sidebar layout with a single-column layout, remove the mode toggle, and tuck settings into a collapsible expander.

**Architecture:** Single-file app stays single-file. Three changes: (1) modify `get_generation_params` signature from `mode: str` to `has_image: bool`, (2) rewrite `main()` UI layout to remove sidebar and mode toggle, (3) update tests for new signature. TDD — update tests first, then implementation.

**Tech Stack:** Streamlit, Python, pytest

---

### Task 1: Update `get_generation_params` signature and tests

**Files:**
- Modify: `streamlit_app.py:40-49`
- Modify: `tests/test_streamlit_app.py:50-81`

- [ ] **Step 1: Update the tests to use `has_image: bool` instead of `mode: str`**

Replace the entire `TestGetGenerationParams` class in `tests/test_streamlit_app.py` with:

```python
class TestGetGenerationParams:
    def test_thinking_mode(self):
        instruction, tokens = get_generation_params(
            has_image=True, is_thinking=True, system_instruction="Be helpful."
        )
        assert (
            instruction == "SYSTEM INSTRUCTION: think silently if needed. Be helpful."
        )
        assert tokens == 1300

    def test_image_non_thinking(self):
        instruction, tokens = get_generation_params(
            has_image=True, is_thinking=False, system_instruction="Be helpful."
        )
        assert instruction == "Be helpful."
        assert tokens == 300

    def test_text_only_non_thinking(self):
        instruction, tokens = get_generation_params(
            has_image=False, is_thinking=False, system_instruction="Be helpful."
        )
        assert instruction == "Be helpful."
        assert tokens == 500

    def test_thinking_text_only(self):
        instruction, tokens = get_generation_params(
            has_image=False, is_thinking=True, system_instruction="Be helpful."
        )
        assert (
            instruction == "SYSTEM INSTRUCTION: think silently if needed. Be helpful."
        )
        assert tokens == 1300
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_streamlit_app.py::TestGetGenerationParams -v`
Expected: FAIL — `get_generation_params()` still expects `mode: str` positional arg, not `has_image` keyword arg.

- [ ] **Step 3: Update `get_generation_params` in `streamlit_app.py`**

Replace the function at lines 40-49 with:

```python
def get_generation_params(
    has_image: bool, is_thinking: bool, system_instruction: str
) -> tuple[str, int]:
    if is_thinking:
        return (
            f"SYSTEM INSTRUCTION: think silently if needed. {system_instruction}",
            1300,
        )
    max_new_tokens = 300 if has_image else 500
    return system_instruction, max_new_tokens
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_streamlit_app.py -v`
Expected: All 10 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add streamlit_app.py tests/test_streamlit_app.py
git commit -m "refactor: change get_generation_params to accept has_image bool instead of mode string"
```

---

### Task 2: Rewrite `main()` layout — remove sidebar, add expander

**Files:**
- Modify: `streamlit_app.py:55-138`

- [ ] **Step 1: Replace the `main()` function in `streamlit_app.py`**

Replace the entire `main()` function (lines 55-138) with:

```python
def main():
    st.title("MedGemma")
    st.caption("Medical image and text analysis powered by MedGemma 1.5 4B")

    with st.spinner("Loading model..."):
        model, processor, config = load_model()

    prompt = st.text_input(
        "Enter your question", placeholder="e.g. Describe this X-ray"
    )

    uploaded_image = None
    uploaded_file = st.file_uploader(
        "Upload a medical image (optional)", type=["png", "jpg", "jpeg", "webp"]
    )
    if uploaded_file is not None:
        try:
            uploaded_image = Image.open(uploaded_file)
            st.image(uploaded_image, caption="Uploaded image", width="stretch")
        except Exception:
            st.error("Failed to load image. Please upload a valid image file.")

    default_instruction = (
        "You are an expert radiologist."
        if uploaded_image is not None
        else "You are a helpful medical assistant."
    )
    with st.expander("Settings"):
        is_thinking = st.toggle("Enable thinking", value=False)
        system_instruction = st.text_area(
            "System instruction", value=default_instruction, height=100
        )

    generate_btn = st.button("Generate", type="primary", disabled=not prompt)

    if generate_btn and prompt:
        has_image = uploaded_image is not None
        full_instruction, max_new_tokens = get_generation_params(
            has_image, is_thinking, system_instruction
        )
        messages = build_messages(prompt, full_instruction, uploaded_image)
        image_for_model = [uploaded_image] if uploaded_image else None
        num_images = 1 if image_for_model else 0
        with st.spinner("Generating response..."):
            try:
                formatted_prompt = apply_chat_template(
                    processor, config, messages, num_images=num_images
                )
                output = generate(
                    model,
                    processor,
                    formatted_prompt,  # ty: ignore[invalid-argument-type]
                    image_for_model,  # ty: ignore[invalid-argument-type]
                    max_tokens=max_new_tokens,
                    temperature=0.0,
                    verbose=False,
                )
                response = output.text
            except Exception as e:
                st.error(f"Inference failed: {e}")
                return

        thought, response = parse_response(response, is_thinking)
        if thought is not None:
            with st.expander("Thinking trace"):
                st.markdown(thought)

        st.markdown("### Response")
        st.markdown(response)
```

- [ ] **Step 2: Run all tests**

Run: `uv run pytest tests/test_streamlit_app.py -v`
Expected: All 10 tests PASS (helper functions unchanged, `main()` not tested directly).

- [ ] **Step 3: Run linter and formatter**

Run: `uv run ruff check . && uv run ruff format .`
Expected: No errors.

- [ ] **Step 4: Run type checker**

Run: `uv run ty check`
Expected: No new errors (existing `ty: ignore` comments preserved).

- [ ] **Step 5: Commit**

```bash
git add streamlit_app.py
git commit -m "refactor: remove sidebar, use single-column layout with settings expander"
```
