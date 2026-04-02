import importlib.util
import pathlib
import types

import pytest
from fastapi import HTTPException


_MODULE_PATH = pathlib.Path(__file__).resolve().parents[1] / "examples" / "openai_server.py"
_SPEC = importlib.util.spec_from_file_location("openai_server", _MODULE_PATH)
openai_server = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(openai_server)


def _set_model_type(model_type: str):
    openai_server.tts_model = types.SimpleNamespace(
        model=types.SimpleNamespace(
            model=types.SimpleNamespace(tts_model_type=model_type),
            get_supported_speakers=lambda: ["aiden", "ava"],
        )
    )


def test_custom_voice_request_maps_voice_to_speaker_and_language_override():
    _set_model_type("custom_voice")
    openai_server.voices = {}
    openai_server.default_voice = "aiden"
    openai_server.default_language = "English"
    openai_server.default_instruct = "speak in a cheerful and energetic tone"

    req = openai_server.SpeechRequest(
        input="hello",
        voice="ava",
        language="Japanese",
        stream=True,
    )
    options = openai_server.resolve_request_options(req)

    assert options == {
        "mode": "custom_voice",
        "speaker": "ava",
        "language": "Japanese",
        "instruct": "speak in a cheerful and energetic tone",
    }


def test_custom_voice_alias_config_can_remap_voice_name():
    _set_model_type("custom_voice")
    openai_server.voices = {
        "alloy": {"speaker": "aiden", "language": "French", "instruct": "calm"}
    }
    openai_server.default_voice = "aiden"
    openai_server.default_language = "English"
    openai_server.default_instruct = "speak in a cheerful and energetic tone"

    req = openai_server.SpeechRequest(input="hello", voice="alloy")
    options = openai_server.resolve_request_options(req)

    assert options == {
        "mode": "custom_voice",
        "speaker": "aiden",
        "language": "French",
        "instruct": "calm",
    }


def test_custom_voice_uses_default_instruction_when_alias_has_none():
    _set_model_type("custom_voice")
    openai_server.voices = {}
    openai_server.default_voice = "aiden"
    openai_server.default_language = "English"
    openai_server.default_instruct = "speak in a cheerful and energetic tone"

    req = openai_server.SpeechRequest(input="hello", voice="aiden")
    options = openai_server.resolve_request_options(req)

    assert options["instruct"] == "speak in a cheerful and energetic tone"


def test_voice_clone_request_uses_profile_and_request_language_override():
    _set_model_type("voice_clone")
    openai_server.voices = {
        "alloy": {
            "ref_audio": "ref.wav",
            "ref_text": "reference",
            "language": "English",
            "chunk_size": 8,
        }
    }
    openai_server.default_voice = "alloy"
    openai_server.default_language = "Auto"

    req = openai_server.SpeechRequest(input="hello", voice="alloy", language="German")
    options = openai_server.resolve_request_options(req)

    assert options == {
        "mode": "voice_clone",
        "language": "German",
        "ref_audio": "ref.wav",
        "ref_text": "reference",
        "chunk_size": 8,
    }


def test_voice_design_uses_voice_text_as_instruction():
    _set_model_type("voice_design")
    openai_server.voices = {}
    openai_server.default_voice = None
    openai_server.default_language = "English"

    req = openai_server.SpeechRequest(
        input="hello",
        voice="Warm, confident narrator with slight British accent",
    )
    options = openai_server.resolve_request_options(req)

    assert options == {
        "mode": "voice_design",
        "language": "English",
        "instruct": "Warm, confident narrator with slight British accent",
    }


def test_voice_design_alias_can_remap_instruction():
    _set_model_type("voice_design")
    openai_server.voices = {
        "narrator": {
            "instruct": "Warm, confident narrator",
            "language": "English",
        }
    }
    openai_server.default_voice = "narrator"
    openai_server.default_language = "Auto"

    req = openai_server.SpeechRequest(input="hello", voice="narrator")
    options = openai_server.resolve_request_options(req)

    assert options == {
        "mode": "voice_design",
        "language": "English",
        "instruct": "Warm, confident narrator",
    }


def test_voice_design_requires_non_empty_instruction_text():
    _set_model_type("voice_design")
    openai_server.voices = {}
    openai_server.default_voice = None
    openai_server.default_language = "English"

    req = openai_server.SpeechRequest(input="hello", voice="")

    with pytest.raises(HTTPException, match="require non-empty 'voice' instruction text"):
        openai_server.resolve_request_options(req)


def test_language_is_detected_from_english_text_when_request_uses_auto():
    _set_model_type("custom_voice")
    openai_server.voices = {}
    openai_server.default_voice = "aiden"
    openai_server.default_language = "Auto"

    req = openai_server.SpeechRequest(
        input="Hello world, this is a fairly normal English sentence.",
        voice="aiden",
        language="Auto",
    )
    options = openai_server.resolve_request_options(req)

    assert options["language"] == "English"


def test_language_is_detected_from_cjk_text():
    _set_model_type("custom_voice")
    openai_server.voices = {}
    openai_server.default_voice = "aiden"
    openai_server.default_language = "Auto"

    req = openai_server.SpeechRequest(
        input="你好，今天过得怎么样？",
        voice="aiden",
    )
    options = openai_server.resolve_request_options(req)

    assert options["language"] == "Chinese"


def test_mixed_input_uses_dominant_language():
    _set_model_type("custom_voice")
    openai_server.voices = {}
    openai_server.default_voice = "aiden"
    openai_server.default_language = "English"

    req = openai_server.SpeechRequest(
        input="Hello 你好, welcome back",
        voice="aiden",
    )
    options = openai_server.resolve_request_options(req)

    assert options["language"] == "English"


def test_mixed_input_can_choose_non_latin_dominant_language():
    _set_model_type("custom_voice")
    openai_server.voices = {}
    openai_server.default_voice = "aiden"
    openai_server.default_language = "English"

    req = openai_server.SpeechRequest(
        input="你好，今天过得怎么样？ hello",
        voice="aiden",
    )
    options = openai_server.resolve_request_options(req)

    assert options["language"] == "Chinese"


def test_explicit_language_is_not_overridden_by_detection():
    _set_model_type("custom_voice")
    openai_server.voices = {}
    openai_server.default_voice = "aiden"
    openai_server.default_language = "Auto"

    req = openai_server.SpeechRequest(
        input="Hello world",
        voice="aiden",
        language="French",
    )
    options = openai_server.resolve_request_options(req)

    assert options["language"] == "French"
