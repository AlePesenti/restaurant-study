"""Text-to-speech engines for the four study conditions.

Robotic voices use eSpeak NG (formant synthesis, the classic "system" voice).
Human-like voices use Kokoro-82M, an open-weight neural TTS model.
Both run locally on the server, so every participant hears exactly the same audio.
"""

import ctypes
import hashlib
import io
import threading
from collections import OrderedDict
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------- eSpeak NG ---

class ESpeak:
    """Minimal ctypes binding to libespeak-ng (bundled by the espeakng_loader package)."""

    SAMPLE_RATE = 22050
    AUDIO_OUTPUT_SYNCHRONOUS = 2
    espeakCHARS_UTF8 = 1
    espeakRATE = 1
    espeakPITCH = 3

    def __init__(self):
        import espeakng_loader

        self.lib = ctypes.cdll.LoadLibrary(espeakng_loader.get_library_path())
        data_path = espeakng_loader.get_data_path().encode("utf-8")

        self.SynthCallback = ctypes.CFUNCTYPE(
            ctypes.c_int, ctypes.POINTER(ctypes.c_short), ctypes.c_int, ctypes.c_void_p)
        self.lib.espeak_Initialize.restype = ctypes.c_int
        self.lib.espeak_Initialize.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_char_p, ctypes.c_int]
        self.lib.espeak_SetSynthCallback.argtypes = [self.SynthCallback]
        self.lib.espeak_SetVoiceByName.argtypes = [ctypes.c_char_p]
        self.lib.espeak_SetParameter.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int]
        self.lib.espeak_Synth.argtypes = [
            ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint, ctypes.c_int, ctypes.c_uint,
            ctypes.c_uint, ctypes.POINTER(ctypes.c_uint), ctypes.c_void_p]

        rate = self.lib.espeak_Initialize(self.AUDIO_OUTPUT_SYNCHRONOUS, 500, data_path, 0)
        if rate > 0:
            self.SAMPLE_RATE = rate

        self._chunks = []

        def on_audio(wav, num_samples, events):
            if num_samples > 0:
                self._chunks.append(np.ctypeslib.as_array(wav, shape=(num_samples,)).copy())
            return 0

        self._callback = self.SynthCallback(on_audio)  # keep a reference alive
        self.lib.espeak_SetSynthCallback(self._callback)

    def synthesize(self, text, voice, rate=150, pitch=None):
        """voice is an eSpeak name plus variant, e.g. "en+m3" (male) or "en+f3" (female)."""
        self._chunks = []
        if self.lib.espeak_SetVoiceByName(voice.encode("utf-8")) != 0:
            raise ValueError(f"eSpeak voice not found: {voice}")
        self.lib.espeak_SetParameter(self.espeakRATE, rate, 0)
        if pitch is not None:
            self.lib.espeak_SetParameter(self.espeakPITCH, pitch, 0)
        data = text.encode("utf-8") + b"\0"
        self.lib.espeak_Synth(data, len(data), 0, 0, 0, self.espeakCHARS_UTF8, None, None)
        audio = np.concatenate(self._chunks) if self._chunks else np.zeros(1, dtype=np.int16)
        return audio.astype(np.float32) / 32768.0, self.SAMPLE_RATE


# ------------------------------------------------------------------- Kokoro ---

class Kokoro:
    SAMPLE_RATE = 24000

    def __init__(self):
        self._pipelines = {}

    def _pipeline(self, lang_code):
        if lang_code not in self._pipelines:
            from kokoro import KPipeline
            self._pipelines[lang_code] = KPipeline(lang_code=lang_code, repo_id="hexgrad/Kokoro-82M")
        return self._pipelines[lang_code]

    def synthesize(self, text, voice, speed=1.0):
        # Kokoro voice names start with the language code: "b" = British, "a" = American.
        pipeline = self._pipeline(voice[0])
        parts = [result.audio.numpy() for result in pipeline(text, voice=voice, speed=speed)]
        audio = np.concatenate(parts) if parts else np.zeros(1, dtype=np.float32)
        return audio, self.SAMPLE_RATE


# --------------------------------------------------------------- front door ---

CACHE_DIR = Path(__file__).resolve().parent / "audio_cache"
SYNTHESIS_KEYS = ("engine", "voice", "rate", "pitch", "speed")


class SpeechSynthesizer:
    """Turns (voice config, text) into MP3 bytes.

    Audio is cached on disk (webapp/audio_cache, filled by webapp/precompute.py) and in
    memory, so a sentence is only ever synthesized once per voice.
    """

    def __init__(self, cache_dir=CACHE_DIR, memory_size=500):
        self._espeak = None
        self._kokoro = None
        # Each engine holds global state, so it synthesizes one sentence at a time. The
        # engines have separate locks so eSpeak never waits for a slow Kokoro sentence.
        self._engine_locks = {"espeak": threading.Lock(), "kokoro": threading.Lock()}
        self._cache_lock = threading.Lock()
        self._memory = OrderedDict()
        self._memory_size = memory_size
        self._cache_dir = Path(cache_dir)

    def _engine(self, name):
        if name == "espeak":
            if self._espeak is None:
                self._espeak = ESpeak()
            return self._espeak
        if name == "kokoro":
            if self._kokoro is None:
                self._kokoro = Kokoro()
            return self._kokoro
        raise ValueError(f"unknown TTS engine: {name}")

    @staticmethod
    def key(voice, text):
        """Engine name + hash, e.g. "kokoro3f2a...". The engine prefix picks the cache folder."""
        settings = sorted((k, v) for k, v in voice.items() if k in SYNTHESIS_KEYS)
        return voice["engine"] + hashlib.sha1(f"{settings}|{text}".encode("utf-8")).hexdigest()[:20]

    def _path(self, key):
        return self._cache_dir / key[:-20] / f"{key}.mp3"

    def on_disk(self, voice, text):
        return self._path(self.key(voice, text)).exists()

    def _remember(self, key, data):
        self._memory[key] = data
        self._memory.move_to_end(key)
        if len(self._memory) > self._memory_size:
            self._memory.popitem(last=False)

    def get(self, key):
        """Return cached MP3 bytes for a key, or None."""
        if not key.isalnum():
            return None
        with self._cache_lock:
            if key in self._memory:
                self._memory.move_to_end(key)
                return self._memory[key]
        path = self._path(key)
        if not path.exists():
            return None
        data = path.read_bytes()
        with self._cache_lock:
            self._remember(key, data)
        return data

    def audio(self, voice, text):
        """Return (cache key, MP3 bytes) for the text spoken with the given voice."""
        key = self.key(voice, text)
        cached = self.get(key)
        if cached is not None:
            return key, cached

        with self._engine_locks[voice["engine"]]:
            cached = self.get(key)   # another request may have rendered it meanwhile
            if cached is not None:
                return key, cached

            options = {k: v for k, v in voice.items() if k in SYNTHESIS_KEYS and k not in ("engine", "voice")}
            samples, sample_rate = self._engine(voice["engine"]).synthesize(text, voice["voice"], **options)

            import soundfile as sf   # only needed when rendering, not to serve the cache

            buffer = io.BytesIO()
            sf.write(buffer, samples, sample_rate, format="MP3", compression_level=0.85)
            data = buffer.getvalue()

            path = self._path(key)
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(".tmp")
            temporary.write_bytes(data)
            temporary.replace(path)   # atomic, so get() never reads a half-written file
            with self._cache_lock:
                self._remember(key, data)

        return key, data
