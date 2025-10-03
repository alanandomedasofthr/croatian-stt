import argparse
import asyncio
import datetime
import importlib
import inspect
import json
import os
import platform
import re
import textwrap
import threading
from dataclasses import dataclass
from queue import Queue
from pathlib import Path
from typing import Any, Callable

import jwt
import numpy as np
import torch
import whisperx
from hunspell import HunSpell

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

_initial_env_keys = set(os.environ.keys())

# ---------- ENV ----------
def load_env_from_file(path: Path, *, overwrite: bool = False) -> None:
    try:
        content = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return
    except OSError as err:  # pragma: no cover - diagnostic only
        print(f"Failed to read env file {path}: {err}")
        return

    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.lower().startswith("export "):
            line = line[7:].lstrip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key:
            continue
        if key in os.environ and (not overwrite or key in _initial_env_keys):
            continue
        value = value.strip()
        if value and value[0] == value[-1] and value[0] in {"\"", "'"}:
            value = value[1:-1]
        os.environ[key] = value


env_dir = Path(__file__).resolve().parent
load_env_from_file(env_dir / ".env")
load_env_from_file(env_dir / ".env.local", overwrite=True)


# ---------- CONFIG ----------
SAMPLE_RATE = 16000        # client sends PCM 16k mono
PROCESS_WINDOW_S = 5       # process every ~5s (tune 2–5)
LOW_CONF_THRESH = 0.7      # word probability threshold
SUB_WRAP_WIDTH = 40
SUB_WRAP_LINES = 2

AUTH_SECRET = os.getenv("AUTH_SECRET", "change-me")  # must match Next.js
HUNSPELL_LANG = os.getenv("HUNSPELL_LANG", "hr_HR")
HUNSPELL_DIR = Path(os.getenv("HUNSPELL_DIR", Path(__file__).parent / "dictionaries")).resolve()
ASR_LANGUAGE = os.getenv("ASR_LANGUAGE", "hr")
ASR_BACKEND = os.getenv("ASR_BACKEND", "auto").lower()
MLX_WHISPER_MODEL = os.getenv("MLX_WHISPER_MODEL", "mlx-community/whisper-large-v3-mlx")
MLX_WHISPER_FP16 = os.getenv("MLX_WHISPER_FP16", "true").lower() != "false"
WHISPERX_MODEL = os.getenv("WHISPERX_MODEL", "large-v3")
WHISPERX_CPU_MODEL = os.getenv("WHISPERX_CPU_MODEL", "large-v2")

# MPS (Apple Silicon) preferred
TORCH_DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = TORCH_DEVICE
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"


def whisper_compute_type(device: str) -> str:
    """Return a compute type compatible with the given device."""
    if device == "cuda":
        return "float16"
    if device == "mps":
        return "float32"
    return "int8"

# ---------- HELPERS ----------
def wrap_subtitle(text: str, width=40, max_lines=2) -> str:
    lines = textwrap.wrap(text, width=width)
    if len(lines) > max_lines:
        lines = lines[:max_lines-1] + [" ".join(lines[max_lines-1:])]
    return "\n".join(lines)

def srt_ts(sec: float) -> str:
    td = datetime.timedelta(seconds=sec)
    s = str(td)
    if "." not in s: s += ".000000"
    h, m, rest = s.split(":")
    s, ms = rest.split(".")
    return f"{int(h):02d}:{int(m):02d}:{int(s):02d},{ms[:3]}"

def vtt_ts(sec: float) -> str:
    td = datetime.timedelta(seconds=sec)
    s = str(td)
    if "." not in s: s += ".000000"
    h, m, rest = s.split(":")
    s, ms = rest.split(".")
    return f"{int(h):02d}:{int(m):02d}:{int(s):02d}.{ms[:3]}"

def verify_token(token: str) -> dict:
    try:
        return jwt.decode(token, AUTH_SECRET, algorithms=["HS256"])
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid token")

def mps_sync(device: str) -> None:
    if device == "mps" and hasattr(torch.mps, "synchronize"):
        torch.mps.synchronize()

# ---------- MODELS ----------
class OpenAIWhisperWrapper:
    """Adapter to expose OpenAI Whisper with the faster-whisper interface."""

    def __init__(self, model: Any, language: str) -> None:
        self._model = model
        self._language = language
        self._transcribe_params = inspect.signature(model.transcribe).parameters

    def transcribe(self, audio: Any, **kwargs: Any) -> dict:
        kwargs.pop("batch_size", None)  # not supported by openai/whisper
        kwargs.setdefault("language", self._language)
        if "task" in self._transcribe_params:
            kwargs.setdefault("task", "transcribe")
        if "translate" in self._transcribe_params:
            kwargs.setdefault("translate", False)
        return self._model.transcribe(audio, **kwargs)


VALID_BACKENDS = {"auto", "mlx", "whisperx"}
if ASR_BACKEND not in VALID_BACKENDS:
    raise ValueError(f"Unsupported ASR_BACKEND '{ASR_BACKEND}'. Expected one of {sorted(VALID_BACKENDS)}")

selected_backend = "whisperx"
mlx_whisper_module: Any | None = None

BackendEntry = tuple[Callable[[np.ndarray], list[dict[str, Any]]], Callable[[], None], str]
backend_cache: dict[tuple[str, ...], BackendEntry] = {}


@dataclass
class TranscriptionJob:
    audio: np.ndarray
    chunk_offset: float
    loop: asyncio.AbstractEventLoop
    future: asyncio.Future


@dataclass
class TranscriptionResult:
    chunk_offset: float
    segments: list[dict[str, Any]]


job_queue: Queue[TranscriptionJob] = Queue()
_worker_started = threading.Event()


def _resolve_future(
    future: asyncio.Future,
    result: TranscriptionResult | None = None,
    exc: Exception | None = None,
) -> None:
    if future.cancelled():
        return
    try:
        if exc is not None:
            future.set_exception(exc)
        else:
            future.set_result(result)
    except asyncio.InvalidStateError:
        pass


def _worker_loop() -> None:
    while True:
        job = job_queue.get()
        try:
            if job.future.cancelled():
                continue
            segments = transcribe_window(job.audio)
            backend_sync()
            result = TranscriptionResult(job.chunk_offset, segments)
            job.loop.call_soon_threadsafe(_resolve_future, job.future, result)
        except Exception as err:  # pragma: no cover - runtime safeguard
            job.loop.call_soon_threadsafe(_resolve_future, job.future, None, err)
        finally:
            job_queue.task_done()


def ensure_worker_started() -> None:
    if _worker_started.is_set():
        return
    worker = threading.Thread(target=_worker_loop, name="asr-worker", daemon=True)
    worker.start()
    _worker_started.set()


_warmup_lock = threading.Lock()
warmed_backends: set[tuple[str, str]] = set()
warming_backends: set[tuple[str, str]] = set()

if ASR_BACKEND in {"auto", "mlx"}:
    try:
        mlx_whisper_module = importlib.import_module("mlx_whisper")  # type: ignore[import-untyped]
    except ModuleNotFoundError as exc:
        if ASR_BACKEND == "mlx":
            raise RuntimeError(
                "ASR_BACKEND=mlx but mlx-whisper is not installed. Run `uv sync` to install dependencies."
            ) from exc
    else:
        if platform.system().lower() != "darwin":
            if ASR_BACKEND == "mlx":
                raise RuntimeError("MLX Whisper backend requires macOS.")
            mlx_whisper_module = None
        else:
            selected_backend = "mlx"


def build_transcribe_kwargs(transcribe_model: Any) -> dict[str, object]:
    """Return kwargs compatible with the provided model.transcribe signature."""

    params = inspect.signature(transcribe_model.transcribe).parameters
    kwargs: dict[str, object] = {"batch_size": 8}
    if "fp16" in params:
        # Disable fp16 to avoid issues on MPS/CPU; CUDA precision is managed internally.
        kwargs["fp16"] = False
    if "language" in params:
        kwargs["language"] = ASR_LANGUAGE
    if "task" in params:
        kwargs["task"] = "transcribe"
    if "translate" in params:
        kwargs["translate"] = False
    return kwargs


def configure_whisperx(initial_device: str) -> BackendEntry:
    cache_key = ("whisperx", initial_device)
    cached = backend_cache.get(cache_key)
    if cached:
        return cached

    device = initial_device
    print("Loading models on:", device)

    try:
        chosen_model = WHISPERX_MODEL if device != "cpu" else WHISPERX_CPU_MODEL
        model = whisperx.load_model(
            chosen_model,
            device,
            compute_type=whisper_compute_type(device),
        )
    except ValueError as exc:
        lowered = str(exc).lower()
        if device == "mps" and "unsupported device" in lowered:
            try:
                import whisper  # type: ignore

                print("WhisperX faster-whisper backend lacks MPS support; using openai/whisper instead")
                openai_model_name = os.getenv("WHISPER_MPS_MODEL", "large-v3")
                whisper_model = whisper.load_model(openai_model_name, device=device)
                model = OpenAIWhisperWrapper(whisper_model, ASR_LANGUAGE)
            except Exception as inner_exc:  # pragma: no cover - informational
                fallback_device = "cuda" if torch.cuda.is_available() else "cpu"
                if fallback_device == device:
                    raise inner_exc from exc
                print(
                    f"Failed to initialize openai/whisper on '{device}': {inner_exc}; "
                    f"falling back to {fallback_device}"
                )
                entry = configure_whisperx(fallback_device)
                backend_cache[cache_key] = entry
                return entry
        elif device != "cpu":
            fallback_device = "cuda" if torch.cuda.is_available() else "cpu"
            if fallback_device == device:
                raise
            print(f"WhisperX unavailable on '{device}'; falling back to {fallback_device}")
            entry = configure_whisperx(fallback_device)
            backend_cache[cache_key] = entry
            return entry
        else:
            raise

    align_model, metadata = whisperx.load_align_model(language_code=ASR_LANGUAGE, device=device)
    print("Models ready on:", device)

    transcribe_kwargs = build_transcribe_kwargs(model)

    def whisperx_transcribe(audio: np.ndarray) -> list[dict[str, Any]]:
        result = model.transcribe(audio, **transcribe_kwargs)
        segments = result.get("segments", [])
        if not segments:
            return []
        aligned = whisperx.align(segments, align_model, metadata, audio, device)
        return aligned.get("segments", []) if aligned else []

    def sync() -> None:
        mps_sync(device)

    entry: BackendEntry = (whisperx_transcribe, sync, device)
    backend_cache[("whisperx", device)] = entry
    if device != initial_device:
        backend_cache[cache_key] = entry
    return entry


transcribe_window: Callable[[np.ndarray], list[dict[str, Any]]]
backend_sync: Callable[[], None]


def switch_to_whisperx() -> None:
    global transcribe_window, backend_sync, selected_backend, DEVICE
    transcribe_window, backend_sync, new_device = configure_whisperx(TORCH_DEVICE)
    selected_backend = "whisperx"
    DEVICE = new_device
    print(f"Switched to WhisperX backend: {new_device}")
    warmup_backend(PROCESS_WINDOW_S)


def warmup_backend(seconds: int = 1) -> None:
    if seconds <= 0:
        return

    backend_key = (selected_backend, DEVICE)
    if backend_key in warmed_backends:
        return

    with _warmup_lock:
        if backend_key in warmed_backends or backend_key in warming_backends:
            return
        warming_backends.add(backend_key)

    print(f"Warming {selected_backend} backend on {DEVICE} with {seconds}s of silence")
    dummy_audio = np.zeros(SAMPLE_RATE * seconds, dtype=np.float32)
    try:
        _ = transcribe_window(dummy_audio)
        backend_sync()
    except Exception as err:  # pragma: no cover - warmup resilience
        print(f"Backend warmup failed: {err}")
        with _warmup_lock:
            warming_backends.discard(backend_key)
        return

    with _warmup_lock:
        warming_backends.discard(backend_key)
        warmed_backends.add(backend_key)
    print("Backend warmup complete")


def configure_mlx(mlx_module: Any) -> BackendEntry:
    cache_key = (
        "mlx",
        MLX_WHISPER_MODEL,
        ASR_LANGUAGE,
        "fp16" if MLX_WHISPER_FP16 else "fp32",
    )
    cached = backend_cache.get(cache_key)
    if cached:
        return cached

    print(f"Loading MLX Whisper model: {MLX_WHISPER_MODEL}")

    mlx_params = inspect.signature(mlx_module.transcribe).parameters

    def build_mlx_kwargs() -> dict[str, object]:
        kwargs: dict[str, object] = {
            "path_or_hf_repo": MLX_WHISPER_MODEL,
            "word_timestamps": True,
        }
        if "fp16" in mlx_params:
            kwargs["fp16"] = MLX_WHISPER_FP16
        if "language" in mlx_params:
            kwargs["language"] = ASR_LANGUAGE
        if "task" in mlx_params:
            kwargs["task"] = "transcribe"
        if "translate" in mlx_params:
            kwargs["translate"] = False
        return kwargs

    mlx_transcribe_kwargs = build_mlx_kwargs()

    def mlx_transcribe(audio: np.ndarray) -> list[dict[str, Any]]:
        try:
            result = mlx_module.transcribe(
                audio,
                **mlx_transcribe_kwargs,
            )
        except Exception as err:  # pragma: no cover - runtime safeguard
            if ASR_BACKEND == "auto":
                print(
                    f"MLX backend failed for model '{MLX_WHISPER_MODEL}': {err}; falling back to WhisperX."
                )
                switch_to_whisperx()
                return transcribe_window(audio)
            raise RuntimeError(
                "Failed to run MLX Whisper transcription. Check your `MLX_WHISPER_MODEL` setting "
                "or authenticate with Hugging Face (`huggingface-cli login`)."
            ) from err

        segments = result.get("segments", []) if result else []
        return segments

    def sync() -> None:
        return None

    entry: BackendEntry = (mlx_transcribe, sync, "mlx")
    backend_cache[cache_key] = entry
    return entry


if selected_backend == "mlx" and mlx_whisper_module is not None:
    transcribe_window, backend_sync, DEVICE = configure_mlx(mlx_whisper_module)
    warmup_backend(PROCESS_WINDOW_S)
else:
    switch_to_whisperx()

print(f"Selected ASR backend: {selected_backend} ({DEVICE})")


def load_hunspell(lang: str, dictionary_dir: Path) -> HunSpell | None:
    dic = dictionary_dir / f"{lang}.dic"
    aff = dictionary_dir / f"{lang}.aff"
    if not dic.exists() or not aff.exists():
        print(f"Hunspell dictionary not found for {lang} in {dictionary_dir}; corrections disabled")
        return None
    try:
        return HunSpell(str(dic), str(aff))
    except OSError as err:
        print(f"Failed to load Hunspell dictionary: {err}; corrections disabled")
        return None


spellchecker = load_hunspell(HUNSPELL_LANG, HUNSPELL_DIR)


def hunspell_correct(text: str) -> str:
    if not spellchecker:
        return text

    def replace_token(token: str) -> str:
        if not token:
            return token
        # Separate leading/trailing punctuation
        match = re.match(r"^([\W_]*)([\wÀ-ÖØ-öø-ÿ]+)([\W_]*)$", token, re.UNICODE)
        if not match:
            return token
        prefix, core, suffix = match.groups()
        core_lower = core.lower()
        if spellchecker.spell(core_lower):
            return token

        suggestions = spellchecker.suggest(core_lower)
        if not suggestions:
            return token

        replacement = suggestions[0]
        if core[0].isupper():
            replacement = replacement.capitalize()
        return f"{prefix}{replacement}{suffix}"

    tokens = re.findall(r"\S+|\s+", text, re.UNICODE)
    corrected_tokens = [replace_token(tok) if not tok.isspace() else tok for tok in tokens]
    return "".join(corrected_tokens)

# ---------- OUTPUT FILES ----------
out_dir = Path("./out"); out_dir.mkdir(exist_ok=True)
RAW_PATH = out_dir/"transcript_raw.txt"
TXT_PATH = out_dir/"transcript_corrected.txt"
SRT_PATH = out_dir/"transcript_corrected.srt"
VTT_PATH = out_dir/"transcript_corrected.vtt"
# reset files
RAW_PATH.write_text("", encoding="utf-8")
TXT_PATH.write_text("", encoding="utf-8")
SRT_PATH.write_text("", encoding="utf-8")
VTT_PATH.write_text("WEBVTT\n\n", encoding="utf-8")

# ---------- APP ----------
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

srt_index = 1
srt_lock = threading.Lock()


async def _send_ws_error(ws: WebSocket, detail: str) -> None:
    payload = {"type": "error", "detail": detail}
    try:
        await ws.send_text(json.dumps(payload))
    except Exception:
        return


async def _stream_transcription(job: TranscriptionJob, ws: WebSocket) -> None:
    try:
        result = await job.future
    except asyncio.CancelledError:  # propagate cancellation while notifying worker
        job.future.cancel()
        raise
    except Exception as err:
        await _send_ws_error(ws, f"Transcription failed: {err}")
        return

    segments = result.segments
    if not segments:
        return

    messages: list[dict[str, Any]] = []
    global srt_index
    with srt_lock:
        chunk_offset = result.chunk_offset
        for seg in segments:
            start = chunk_offset + float(seg.get("start", 0.0))
            end = chunk_offset + float(seg.get("end", 0.0))

            words_info = seg.get("words") or []
            if not words_info:
                continue

            words = []
            for w in words_info:
                word = w.get("word")
                if not word:
                    continue
                conf = float(w.get("probability", 1.0))
                words.append(f"[?{word}?]" if conf < LOW_CONF_THRESH else word)

            raw_text = " ".join(words).strip()
            if not raw_text:
                continue

            corrected = hunspell_correct(raw_text)
            corrected_wrapped = wrap_subtitle(
                corrected,
                width=SUB_WRAP_WIDTH,
                max_lines=SUB_WRAP_LINES,
            )

            plain_ts = str(datetime.timedelta(seconds=int(start)))
            with RAW_PATH.open("a", encoding="utf-8") as f:
                f.write(f"[{plain_ts}] {raw_text}\n")
            with TXT_PATH.open("a", encoding="utf-8") as f:
                f.write(f"[{plain_ts}] {corrected}\n")
            with SRT_PATH.open("a", encoding="utf-8") as f:
                f.write(f"{srt_index}\n{srt_ts(start)} --> {srt_ts(end)}\n{corrected_wrapped}\n\n")
            with VTT_PATH.open("a", encoding="utf-8") as f:
                f.write(f"{vtt_ts(start)} --> {vtt_ts(end)}\n{corrected_wrapped}\n\n")

            messages.append({
                "type": "segment",
                "srtIndex": srt_index,
                "start": start,
                "end": end,
                "raw": raw_text,
                "corrected": corrected,
                "correctedWrapped": corrected_wrapped,
            })
            srt_index += 1

    for message in messages:
        try:
            await ws.send_text(json.dumps(message))
        except Exception:
            break


@app.websocket("/ws")
async def ws_handler(ws: WebSocket):
    # JWT from query
    token = ws.query_params.get("token")
    if not token:
        await ws.close(code=4401)
        return
    try:
        _claims = verify_token(token)
    except HTTPException:
        await ws.close(code=4401)
        return

    await ws.accept()
    ensure_worker_started()

    loop = asyncio.get_running_loop()
    pcm_buffer = np.zeros(0, dtype=np.int16)
    processed_samples = 0
    pending_tasks: set[asyncio.Task[Any]] = set()

    try:
        while True:
            # Receive little-endian int16 PCM frames (ArrayBuffer)
            data = await ws.receive_bytes()
            chunk = np.frombuffer(data, dtype=np.int16)
            pcm_buffer = np.concatenate([pcm_buffer, chunk])

            need = SAMPLE_RATE * PROCESS_WINDOW_S
            while len(pcm_buffer) >= need:
                # Take first window, keep remainder
                window = pcm_buffer[:need].astype(np.float32) / 32768.0
                pcm_buffer = pcm_buffer[need:]
                chunk_offset = processed_samples / SAMPLE_RATE
                processed_samples += need

                future = loop.create_future()
                job = TranscriptionJob(
                    audio=window,
                    chunk_offset=chunk_offset,
                    loop=loop,
                    future=future,
                )
                job_queue.put(job)

                task = asyncio.create_task(_stream_transcription(job, ws))
                pending_tasks.add(task)
                task.add_done_callback(pending_tasks.discard)

    except WebSocketDisconnect:
        pass
    finally:
        if pending_tasks:
            for task in list(pending_tasks):
                if not task.done():
                    task.cancel()
            await asyncio.gather(*pending_tasks, return_exceptions=True)

# downloads
@app.get("/download/raw")
def dl_raw() -> FileResponse:
    return FileResponse(RAW_PATH)


@app.get("/download/corrected")
def dl_txt() -> FileResponse:
    return FileResponse(TXT_PATH)


@app.get("/download/srt")
def dl_srt() -> FileResponse:
    return FileResponse(SRT_PATH)


@app.get("/download/vtt")
def dl_vtt() -> FileResponse:
    return FileResponse(VTT_PATH)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Croatian Speech-to-Text Backend")
    parser.add_argument("--port", "-p", type=int, default=7860, help="Port to run the server on (default: 7860)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to (default: 0.0.0.0)")
    args = parser.parse_args()
    
    print(f"Starting server on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)
