#!/usr/bin/env python3
"""
OpenAI-compatible TTS API server for faster-qwen3-tts.

Exposes POST /v1/audio/speech compatible with OpenAI's TTS API, enabling
integration with OpenWebUI, llama-swap, and other OpenAI-compatible clients.

Usage:
    pip install "faster-qwen3-tts[demo]"

    # Single default voice:
    python examples/openai_server.py \\
        --ref-audio voice.wav --ref-text "Reference transcription" \\
        --language English

    # Multiple named voices from a JSON config:
    python examples/openai_server.py --voices voices.json

    # Custom model and port:
    python examples/openai_server.py \\
        --model Qwen/Qwen3-TTS-12Hz-0.6B-Base \\
        --ref-audio voice.wav --ref-text "transcript" \\
        --port 8000

Voices config (voices.json):
    {
        "alloy": {"ref_audio": "voice.wav", "ref_text": "...", "language": "English"},
        "echo":  {"ref_audio": "voice2.wav", "ref_text": "...", "language": "English"}
    }

API usage:
    curl -s http://localhost:8000/v1/audio/speech \\
        -H "Content-Type: application/json" \\
        -d '{"model": "tts-1", "input": "Hello!", "voice": "alloy", "response_format": "wav"}' \\
        --output speech.wav
"""
import argparse
import asyncio
import io
import json
import logging
import os
import queue
import re
import struct
import sys
import threading
from typing import AsyncGenerator, Callable, Optional

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------

app = FastAPI(title="faster-qwen3-tts OpenAI-compatible API")

tts_model = None
voices: dict = {}
default_voice: Optional[str] = None
default_language: str = "Auto"
SAMPLE_RATE = 24000  # updated once the model loads
_model_lock = threading.Lock()  # prevent concurrent GPU inference
_STREAM_QUEUE_MAXSIZE = 4

# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class SpeechRequest(BaseModel):
    model: str = "tts-1"
    input: str
    voice: str = "alloy"
    response_format: str = "wav"  # wav | pcm | mp3
    speed: float = 1.0           # accepted but not yet applied
    language: Optional[str] = None
    stream: bool = False


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------


def _to_pcm16(pcm: np.ndarray) -> bytes:
    """Convert float32 numpy array to raw 16-bit little-endian PCM bytes."""
    return np.clip(pcm * 32768, -32768, 32767).astype(np.int16).tobytes()


def _wav_header(sample_rate: int, data_len: int = 0xFFFFFFFF) -> bytes:
    """Build a WAV header.  Use data_len=0xFFFFFFFF for streaming (unknown size)."""
    n_channels = 1
    bits = 16
    byte_rate = sample_rate * n_channels * bits // 8
    block_align = n_channels * bits // 8
    riff_size = 0xFFFFFFFF if data_len == 0xFFFFFFFF else 36 + data_len
    buf = io.BytesIO()
    buf.write(b"RIFF")
    buf.write(struct.pack("<I", riff_size))
    buf.write(b"WAVE")
    buf.write(b"fmt ")
    buf.write(struct.pack("<IHHIIHH", 16, 1, n_channels, sample_rate,
                          byte_rate, block_align, bits))
    buf.write(b"data")
    buf.write(struct.pack("<I", data_len))
    return buf.getvalue()


def _to_wav_bytes(pcm: np.ndarray, sample_rate: int) -> bytes:
    """Convert float32 numpy array to a complete WAV file in memory."""
    raw = _to_pcm16(pcm)
    return _wav_header(sample_rate, len(raw)) + raw


def _to_mp3_bytes(pcm: np.ndarray, sample_rate: int) -> bytes:
    """Convert float32 numpy array to MP3 bytes (requires pydub + ffmpeg)."""
    try:
        from pydub import AudioSegment
    except ImportError:
        raise HTTPException(
            status_code=400,
            detail="response_format='mp3' requires pydub: pip install pydub",
        )
    segment = AudioSegment(
        _to_pcm16(pcm),
        frame_rate=sample_rate,
        sample_width=2,
        channels=1,
    )
    buf = io.BytesIO()
    segment.export(buf, format="mp3")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Voice resolution
# ---------------------------------------------------------------------------


def resolve_voice(voice_name: str) -> dict:
    """Return voice config dict or fall back to default, else raise 400."""
    if voice_name in voices:
        return voices[voice_name]
    if default_voice and default_voice in voices:
        logger.warning(
            "Voice %r not configured; falling back to default voice %r",
            voice_name,
            default_voice,
        )
        return voices[default_voice]
    raise HTTPException(
        status_code=400,
        detail=(
            f"Voice {voice_name!r} is not configured. "
            f"Available voices: {list(voices.keys())}"
        ),
    )


def get_loaded_model_type() -> str:
    """Return the current model family exposed by qwen-tts."""
    if tts_model is None:
        return "unknown"
    return getattr(getattr(tts_model.model, "model", None), "tts_model_type", "voice_clone")


def get_supported_speakers() -> list[str]:
    """Best-effort list of supported CustomVoice speaker IDs."""
    if tts_model is None:
        return []
    getter = getattr(tts_model.model, "get_supported_speakers", None)
    if getter is None:
        return []
    try:
        return getter() or []
    except Exception:
        return []


def detect_language_from_text(text: str) -> Optional[str]:
    """
    Best-effort language detection for request text.

    This is intentionally conservative. It only returns a language when the
    script strongly suggests one; otherwise it returns None and the caller can
    fall back to Auto/default behavior.
    """
    if not text:
        return None

    stripped = text.strip()
    if len(stripped) < 2:
        return None

    if re.search(r"[\u3040-\u30ff]", stripped):
        return "Japanese"
    if re.search(r"[\uac00-\ud7af]", stripped):
        return "Korean"
    if re.search(r"[\u4e00-\u9fff]", stripped):
        return "Chinese"
    if re.search(r"[\u0400-\u04ff]", stripped):
        return "Russian"
    if re.search(r"[\u0600-\u06ff]", stripped):
        return "Arabic"
    if re.search(r"[\u0e00-\u0e7f]", stripped):
        return "Thai"

    latin_words = re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", stripped)
    if latin_words:
        alpha_chars = sum(ch.isalpha() for ch in stripped)
        latin_chars = sum(ch.isascii() and ch.isalpha() for ch in stripped)
        if alpha_chars > 0 and latin_chars / alpha_chars >= 0.8 and len("".join(latin_words)) >= 6:
            return "English"

    return None


def resolve_request_options(req: SpeechRequest) -> dict:
    """
    Map an OpenAI-compatible request onto the loaded Qwen3-TTS model surface.

    CustomVoice:
        `voice` maps to speaker ID, unless --voices defines aliases.
    Voice clone:
        `voice` maps to a configured reference-audio profile.
    Voice design:
        `voice` maps to a configured instruction profile.
    """
    model_type = get_loaded_model_type()
    requested_voice = (req.voice or "").strip()
    alias_cfg = voices.get(requested_voice, {})
    requested_language = req.language
    if requested_language and requested_language.lower() != "auto":
        language = requested_language
    else:
        language = (
            detect_language_from_text(req.input)
            or alias_cfg.get("language")
            or default_language
            or "Auto"
        )

    if model_type == "custom_voice":
        supported = {speaker.lower(): speaker for speaker in get_supported_speakers()}
        if alias_cfg.get("speaker"):
            speaker = alias_cfg["speaker"]
        elif requested_voice.lower() in supported:
            speaker = supported[requested_voice.lower()]
        elif default_voice:
            logger.warning(
                "Speaker %r not available; falling back to default speaker %r",
                requested_voice,
                default_voice,
            )
            speaker = default_voice
        else:
            speaker = requested_voice
        if not speaker:
            raise HTTPException(
                status_code=400,
                detail="CustomVoice requests require a non-empty 'voice' speaker ID",
            )
        return {
            "mode": "custom_voice",
            "speaker": speaker,
            "language": language,
            "instruct": alias_cfg.get("instruct"),
        }

    if model_type == "voice_design":
        if not alias_cfg:
            raise HTTPException(
                status_code=400,
                detail=(
                    "VoiceDesign models require --voices config entries with "
                    "{instruct, language}. Requested voice was not configured."
                ),
            )
        instruct = alias_cfg.get("instruct")
        if not instruct:
            raise HTTPException(
                status_code=400,
                detail=f"Voice {requested_voice!r} is missing required 'instruct' config",
            )
        return {
            "mode": "voice_design",
            "language": language,
            "instruct": instruct,
        }

    voice_cfg = resolve_voice(requested_voice)
    return {
        "mode": "voice_clone",
        "language": req.language or voice_cfg.get("language") or default_language or "Auto",
        "ref_audio": voice_cfg["ref_audio"],
        "ref_text": voice_cfg.get("ref_text", ""),
        "chunk_size": voice_cfg.get("chunk_size", 12),
    }


def run_non_streaming_generation(options: dict, text: str):
    """Generate full audio for the active model family."""
    with _model_lock:
        if options["mode"] == "custom_voice":
            return tts_model.generate_custom_voice(
                text=text,
                speaker=options["speaker"],
                language=options["language"],
                instruct=options.get("instruct"),
            )
        if options["mode"] == "voice_design":
            return tts_model.generate_voice_design(
                text=text,
                instruct=options["instruct"],
                language=options["language"],
            )
        return tts_model.generate_voice_clone(
            text=text,
            language=options["language"],
            ref_audio=options["ref_audio"],
            ref_text=options.get("ref_text", ""),
        )


# ---------------------------------------------------------------------------
# Streaming helper: run sync generator in a background thread
# ---------------------------------------------------------------------------


async def _stream_chunks(
    options: dict,
    text: str,
    stop_event: threading.Event,
) -> AsyncGenerator[bytes, None]:
    """
    Run the appropriate streaming generator in a background thread and yield
    raw PCM bytes for each chunk as they arrive.
    """
    q: queue.Queue = queue.Queue(maxsize=_STREAM_QUEUE_MAXSIZE)
    _DONE = object()

    def _put_until_stopped(item) -> bool:
        while not stop_event.is_set():
            try:
                q.put(item, timeout=0.1)
                return True
            except queue.Full:
                continue
        return False

    def producer():
        try:
            with _model_lock:
                stream_fn: Callable
                kwargs: dict
                if options["mode"] == "custom_voice":
                    stream_fn = tts_model.generate_custom_voice_streaming
                    kwargs = {
                        "text": text,
                        "speaker": options["speaker"],
                        "language": options["language"],
                        "instruct": options.get("instruct"),
                        "non_streaming_mode": False,
                        "chunk_size": options.get("chunk_size", 12),
                    }
                elif options["mode"] == "voice_design":
                    stream_fn = tts_model.generate_voice_design_streaming
                    kwargs = {
                        "text": text,
                        "instruct": options["instruct"],
                        "language": options["language"],
                        "non_streaming_mode": False,
                        "chunk_size": options.get("chunk_size", 12),
                    }
                else:
                    stream_fn = tts_model.generate_voice_clone_streaming
                    kwargs = {
                        "text": text,
                        "language": options["language"],
                        "ref_audio": options["ref_audio"],
                        "ref_text": options.get("ref_text", ""),
                        "chunk_size": options.get("chunk_size", 12),
                        "non_streaming_mode": False,
                    }

                for chunk, _sr, _timing in stream_fn(**kwargs):
                    if stop_event.is_set():
                        break
                    if not _put_until_stopped(chunk):
                        break
        except Exception as exc:
            _put_until_stopped(exc)
        finally:
            _put_until_stopped(_DONE)

    thread = threading.Thread(target=producer, daemon=True)
    thread.start()

    loop = asyncio.get_event_loop()
    try:
        while True:
            item = await loop.run_in_executor(None, q.get)
            if item is _DONE:
                break
            if isinstance(item, Exception):
                raise item
            yield _to_pcm16(item)
    finally:
        stop_event.set()


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": tts_model is not None}


@app.post("/v1/audio/speech")
async def create_speech(req: SpeechRequest, request: Request):
    if tts_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if not req.input.strip():
        raise HTTPException(status_code=400, detail="'input' text is empty")
    fmt = req.response_format.lower()
    options = resolve_request_options(req)

    _CONTENT_TYPES = {
        "wav": "audio/wav",
        "pcm": "audio/pcm",
        "mp3": "audio/mpeg",
    }
    if fmt not in _CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"response_format {fmt!r} not supported. Use: wav, pcm, mp3",
        )
    content_type = _CONTENT_TYPES[fmt]
    if req.stream and fmt == "mp3":
        raise HTTPException(
            status_code=400,
            detail="response_format='mp3' does not support streaming; use wav or pcm",
        )

    # --- Non-streaming: generate all audio and return one payload ---
    if not req.stream:
        loop = asyncio.get_event_loop()
        audio_arrays, sr = await loop.run_in_executor(
            None,
            lambda: run_non_streaming_generation(options, req.input),
        )
        audio = audio_arrays[0] if audio_arrays else np.zeros(1, dtype=np.float32)
        if fmt == "mp3":
            body = _to_mp3_bytes(audio, sr)
        elif fmt == "wav":
            body = _to_wav_bytes(audio, sr)
        else:
            body = _to_pcm16(audio)
        return Response(content=body, media_type=content_type)

    # --- WAV / PCM: stream chunks as they are generated ---
    async def audio_stream():
        stop_event = threading.Event()
        if fmt == "wav":
            yield _wav_header(SAMPLE_RATE)  # stream with unknown data length
        try:
            async for raw_chunk in _stream_chunks(options, req.input, stop_event):
                if await request.is_disconnected():
                    logger.info("Streaming client disconnected; stopping generation")
                    stop_event.set()
                    break
                yield raw_chunk
                if await request.is_disconnected():
                    logger.info("Streaming client disconnected after chunk; stopping generation")
                    stop_event.set()
                    break
        finally:
            stop_event.set()

    return StreamingResponse(audio_stream(), media_type=content_type)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _parse_args():
    p = argparse.ArgumentParser(
        description="OpenAI-compatible TTS server for faster-qwen3-tts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--model",
        default=os.environ.get("QWEN_TTS_MODEL", "Qwen/Qwen3-TTS-12Hz-1.7B-Base"),
        help="HuggingFace model ID or local path (default: Qwen/Qwen3-TTS-12Hz-1.7B-Base)",
    )
    p.add_argument(
        "--voices",
        default=os.environ.get("QWEN_TTS_VOICES"),
        metavar="FILE",
        help="JSON file mapping voice names to {ref_audio, ref_text, language}",
    )
    p.add_argument(
        "--ref-audio",
        default=os.environ.get("QWEN_TTS_REF_AUDIO"),
        metavar="FILE",
        help="Reference audio file when --voices is not used",
    )
    p.add_argument(
        "--ref-text",
        default=os.environ.get("QWEN_TTS_REF_TEXT", ""),
        help="Transcript of --ref-audio",
    )
    p.add_argument(
        "--language",
        default=os.environ.get("QWEN_TTS_LANGUAGE", "Auto"),
        help="Target language (English, French, Auto, …) when --voices is not used",
    )
    p.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    p.add_argument("--port", type=int, default=8000, help="Bind port (default: 8000)")
    p.add_argument("--device", default="cuda", help="Torch device (default: cuda)")
    return p.parse_args()


def main():
    global tts_model, voices, default_voice, default_language, SAMPLE_RATE

    args = _parse_args()
    default_language = args.language

    # Build voice registry
    if args.voices:
        with open(args.voices) as f:
            voices = json.load(f)
        default_voice = next(iter(voices))
        logger.info("Loaded %d voice(s) from %s", len(voices), args.voices)
    elif args.ref_audio:
        voices = {
            "default": {
                "ref_audio": args.ref_audio,
                "ref_text": args.ref_text,
                "language": args.language,
            }
        }
        default_voice = "default"
        logger.info("Using single voice from --ref-audio: %s", args.ref_audio)
    else:
        print(
            "ERROR: provide --ref-audio <file> or --voices <config.json>",
            file=sys.stderr,
        )
        sys.exit(1)

    from faster_qwen3_tts import FasterQwen3TTS

    logger.info("Loading model %s on %s …", args.model, args.device)
    tts_model = FasterQwen3TTS.from_pretrained(
        args.model,
        device=args.device,
        dtype=torch.bfloat16,
    )
    SAMPLE_RATE = tts_model.sample_rate
    if get_loaded_model_type() == "custom_voice" and not voices:
        speakers = get_supported_speakers()
        if speakers:
            default_voice = speakers[0]
            logger.info(
                "CustomVoice model detected; requests can use speaker IDs directly via "
                "'voice'. Default speaker: %s",
                default_voice,
            )
    logger.info("Model ready. Sample rate: %d Hz", SAMPLE_RATE)
    logger.info("Server listening on http://%s:%d", args.host, args.port)

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
