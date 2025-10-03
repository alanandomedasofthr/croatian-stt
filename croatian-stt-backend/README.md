## ASR Backends

- Set `ASR_BACKEND=mlx` to use the MLX Whisper pipeline on macOS. Models default to `mlx-community/whisper-large-v3`; override via `MLX_WHISPER_MODEL`. Toggle dtype with `MLX_WHISPER_FP16=false` if you hit precision issues.
- Some larger MLX models (e.g. `whisper-large-v3`) may require a Hugging Face login. Run `huggingface-cli login` or set `HUGGINGFACE_HUB_TOKEN` before starting the backend if downloads fail.
- Leave `ASR_BACKEND` unset (or `auto`) to prefer MLX when available and fall back to WhisperX otherwise.
- Set `ASR_BACKEND=whisperx` to force the existing WhisperX + alignment path. Override model names with `WHISPERX_MODEL` (default `large-v3`) and `WHISPERX_CPU_MODEL` (default `large-v2`).
