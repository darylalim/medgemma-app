# Remove Sidebar — UI Redesign

## Goal

Replace the sidebar layout with a cleaner single-column layout. Remove the mode toggle by making image upload always available but optional.

## Current State

- Sidebar contains: mode radio (Image+Text / Text Only), thinking toggle, system instruction text area
- Mode toggle controls whether the image uploader is shown
- `get_generation_params()` takes a `mode` string to determine `max_new_tokens`

## Proposed Changes

### UI Layout (top to bottom)

1. Title + caption
2. Text input — "Enter your question"
3. Image uploader — always visible, labeled "(optional)"
4. Settings expander (`st.expander("Settings")`) containing:
   - Thinking toggle (`st.toggle`, default off)
   - System instruction text area (default adjusts based on whether an image was uploaded)
5. Generate button (primary)
6. Response area — thinking trace expander (if enabled) + markdown response

### Logic Changes

- **Remove** `with st.sidebar:` block and mode radio
- **Remove** all references to the `mode` variable
- **Modify `get_generation_params()`**: replace `mode: str` parameter with `has_image: bool`. Token logic: 300 with image, 500 without.
- **Default system instruction**: "You are an expert radiologist." when an image is uploaded, "You are a helpful medical assistant." otherwise
- **Image upload handling**: always render the uploader; pass `uploaded_image` to `build_messages` directly (no mode check)
- **Generate button guard**: remove the warning about missing image in Image+Text mode (no longer applicable)

### Test Changes

- `TestGetGenerationParams`: update all tests to pass `has_image: bool` instead of `mode: str`
- No other test changes needed — `parse_response` and `build_messages` are unaffected

### Files Modified

- `streamlit_app.py` — all UI and logic changes
- `tests/test_streamlit_app.py` — update `get_generation_params` tests
