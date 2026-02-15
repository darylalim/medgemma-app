from PIL import Image

from streamlit_app import build_messages, get_generation_params, parse_response


class TestParseResponse:
    def test_plain_response(self):
        thought, answer = parse_response("Normal answer text", is_thinking=False)
        assert thought is None
        assert answer == "Normal answer text"

    def test_thinking_with_markers(self):
        raw = "<unused94>thought\nSome reasoning here<unused95>Final answer"
        thought, answer = parse_response(raw, is_thinking=True)
        assert thought == "Some reasoning here"
        assert answer == "Final answer"

    def test_thinking_enabled_no_markers(self):
        thought, answer = parse_response("Just a plain reply", is_thinking=True)
        assert thought is None
        assert answer == "Just a plain reply"

    def test_thinking_missing_prefix(self):
        raw = "Some reasoning<unused95>Final answer"
        thought, answer = parse_response(raw, is_thinking=True)
        assert thought == "Some reasoning"
        assert answer == "Final answer"


class TestBuildMessages:
    def test_text_only(self):
        msgs = build_messages("What is a fracture?", "You are a doctor.", image=None)
        assert len(msgs) == 2
        assert msgs[0]["role"] == "system"
        assert msgs[0]["content"] == [{"type": "text", "text": "You are a doctor."}]
        assert msgs[1]["role"] == "user"
        assert msgs[1]["content"] == [{"type": "text", "text": "What is a fracture?"}]

    def test_with_image(self):
        img = Image.new("RGB", (10, 10))
        msgs = build_messages("Describe this", "You are a radiologist.", image=img)
        assert len(msgs) == 2
        user_content = msgs[1]["content"]
        assert len(user_content) == 2
        assert user_content[0] == {"type": "text", "text": "Describe this"}
        assert user_content[1]["type"] == "image"
        assert user_content[1]["image"] is img


class TestGetGenerationParams:
    def test_thinking_mode(self):
        instruction, tokens = get_generation_params(
            "Image + Text", is_thinking=True, system_instruction="Be helpful."
        )
        assert (
            instruction == "SYSTEM INSTRUCTION: think silently if needed. Be helpful."
        )
        assert tokens == 1300

    def test_image_text_non_thinking(self):
        instruction, tokens = get_generation_params(
            "Image + Text", is_thinking=False, system_instruction="Be helpful."
        )
        assert instruction == "Be helpful."
        assert tokens == 300

    def test_text_only_non_thinking(self):
        instruction, tokens = get_generation_params(
            "Text Only", is_thinking=False, system_instruction="Be helpful."
        )
        assert instruction == "Be helpful."
        assert tokens == 500

    def test_thinking_text_only_mode(self):
        instruction, tokens = get_generation_params(
            "Text Only", is_thinking=True, system_instruction="Be helpful."
        )
        assert (
            instruction == "SYSTEM INSTRUCTION: think silently if needed. Be helpful."
        )
        assert tokens == 1300
