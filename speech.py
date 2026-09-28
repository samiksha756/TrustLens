"""
TrustLens — Speech-to-Text Module (Model 4, AUDIO data-space)
--------------------------------------------------------------
Voice-note scams ("Hi, this is your bank's fraud team, we need to verify...")
are increasingly common and completely invisible to a text-only tool. This
module transcribes an uploaded audio clip so the transcript can re-enter the
shared text pipeline, giving TrustLens a third data-space (audio) alongside
text and image.

Engine: OpenAI Whisper (`openai-whisper`, pip). Whisper is a pre-trained
automatic-speech-recognition model. The "base" size is a reasonable balance of
accuracy and speed on a laptop CPU.

Graceful degradation: Whisper + ffmpeg are heavy dependencies. If they are not
installed, this module returns available=False with guidance instead of raising,
so the rest of TrustLens keeps working. This lets the audio route be a genuine,
integrated part of the architecture while remaining optional to install.
"""

import os
import tempfile
from functools import lru_cache

_WHISPER_SIZE = "base"


@lru_cache(maxsize=1)
def _load_model():
    import whisper  # heavy import, done lazily
    return whisper.load_model(_WHISPER_SIZE)


def _engine_ready() -> bool:
    try:
        import whisper  # noqa: F401
        return True
    except Exception:  # noqa: BLE001
        return False


def transcribe_audio(audio_bytes: bytes, suffix: str = ".wav") -> dict:
    """
    Transcribe raw audio bytes to text.

    Returns
    -------
    dict with keys: available (bool), text (str), note (str).
    """
    if not _engine_ready():
        return {
            "available": False,
            "text": "",
            "note": ("Speech engine not installed. `pip install openai-whisper` and "
                     "install ffmpeg to enable voice-note analysis."),
        }

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        model = _load_model()
        result = model.transcribe(tmp_path, fp16=False)
        text = (result.get("text") or "").strip()
        return {
            "available": True,
            "text": text,
            "note": "Transcription succeeded." if text else "Audio contained no detectable speech.",
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "available": False,
            "text": "",
            "note": f"Transcription failed: {type(exc).__name__}: {exc}",
        }
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


if __name__ == "__main__":
    print(transcribe_audio(b"", suffix=".wav"))  # exercises the fallback path
