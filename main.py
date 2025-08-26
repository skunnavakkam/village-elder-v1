#!/usr/bin/env python3
"""
Minimal wake word listener with epaper display and single-shot transcription.
Changes from streaming version:
- No partial/real-time transcription
- Buffer entire session's audio
- Transcribe once after 40s OR after 5s of silence
- Same wake hysteresis, cooldowns, and wake model "virginize" reset
"""

import os
import time
import threading
from collections import deque
from typing import Deque, Optional

import numpy as np
import sounddevice as sd
from openwakeword.model import Model as WakeModel
from faster_whisper import WhisperModel
import epaper
from display import TextBufferEPD
from printertest import print_markdown
from openai import OpenAI
from dotenv import load_dotenv
import datetime

# ================================
# Configuration constants
# ================================
SAMPLE_RATE = 16000            # Hz
CHANNELS = 1
DTYPE = "int16"
CHUNK_SIZE = 1280              # samples per callback (~80ms at 16kHz)

WAKE_SCORE_THRESHOLD = 0.50    # wake score needed to trigger
WAKE_HYSTERESIS_FRAMES = 3     # require N consecutive frames over threshold
RETRIGGER_GAP_SEC = 1.5        # min time between triggers (guards bounce)
POST_SESSION_COOLDOWN_SEC = 1.5  # ignore wakeword for N sec after a session

SESSION_MAX_SEC = 40.0         # hard stop after N seconds
SILENCE_TIMEOUT_SEC = 5.0      # stop after N seconds of silence
RMS_SILENCE_THRESHOLD = 500   # RMS threshold for silence (tune per mic/room)

# ================================
# OpenAI client setup
# ================================
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_reasoning_difficulty() -> str:
    return "low"

def get_gpt_5_answer(text: str) -> str:
    response = client.responses.create(
        model="gpt-5-mini",
        input=[
            {
                "role": "developer",
                "content": "Respond concisely in under a quarter page. Your response will be printed out."
            }, 
            {
                "role": "user",
                "content": text
            }
        ],
        tools=[{"type": "web_search_preview"}, {"type": "code_interpreter", "container": {"type": "auto"}}],
        reasoning={"effort": get_reasoning_difficulty()},
        text={"verbosity": "low"},
    )
    return response.output_text

# ================================
# Utility
# ================================
def is_silence(audio_chunk: np.ndarray, threshold: float = RMS_SILENCE_THRESHOLD) -> bool:
    """RMS-based silence detector."""
    if audio_chunk.size == 0:
        return True
    rms = np.sqrt(np.mean(audio_chunk.astype(np.float32) ** 2))
    print(rms)
    return rms < threshold

# ================================
# Main
# ================================
def main() -> None:
    # ---- E-paper init ----
    epd = epaper.epaper("epd2in13_V4").EPD()
    epd.init()
    text_buffer = TextBufferEPD(epd, font_path="/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", font_size=16)
    text_buffer.clear()

    print("Loading wake word model...")
    oww = WakeModel()

    # Pick a target wakeword (prefer jarvis; else first available)
    keys = list(oww.models.keys())
    if not keys:
        raise RuntimeError("No wakeword models found!")
    target_key = None
    for k in keys:
        lk = k.lower()
        if "hey_jarvis" in lk or "jarvis" in lk:
            target_key = k
            break
    if target_key is None:
        target_key = keys[0]

    print(f"Listening for: {target_key}")

    # ---- Whisper init ----
    print("Loading whisper model...")
    whisper_model = WhisperModel("tiny.en", device="cpu", compute_type="int8")
    print("Ready! Press Ctrl+C to stop.")

    text_buffer.set_text(f"Ready! {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # ---- Sounddevice defaults ----
    sd.default.channels = CHANNELS
    sd.default.samplerate = SAMPLE_RATE
    sd.default.dtype = DTYPE

    # ---- State ----
    last_trigger_ts: float = 0.0
    transcribing: bool = False

    wake_cooldown_until: float = 0.0   # post-session cooldown
    consecutive_wake_hits: int = 0     # hysteresis accumulator

    transcription_buffer: Deque[np.int16] = deque()
    transcription_thread: Optional[threading.Thread] = None

    transcription_start_ts: float = 0.0
    silence_start_ts: float = 0.0

    state_lock = threading.Lock()

    # --------------------------
    # Helpers
    # --------------------------
    def virginize_wake_model() -> None:
        """Reset openwakeword to a 'fresh' state after a session."""
        nonlocal oww
        try:
            oww.reset()
            print("🧹 Wake model reset via reset()")
            text_buffer.set_text(f"Ready! {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        except AttributeError:
            try:
                oww = WakeModel()
                print("🧹 Wake model re-instantiated")
            except Exception as e:
                print(f"⚠️ Failed to re-instantiate wake model: {e}")

    def end_session_and_postprocess() -> None:
        """Stop transcription, do a single decode on the full buffer, send to LLM, cooldown, and reset wake model."""
        nonlocal transcribing, wake_cooldown_until

        # Grab all remaining audio
        with state_lock:
            total_len = len(transcription_buffer)
            if total_len == 0:
                print("\n✅ Session ended (no audio captured)")
            else:
                print(f"\n✅ Session ended with {total_len / SAMPLE_RATE:.2f}s of audio")

            # Drain buffer into contiguous array
            if total_len > 0:
                arr = np.empty(total_len, dtype=np.int16)
                for i in range(total_len):
                    arr[i] = transcription_buffer.popleft()
            else:
                arr = None

        transcript_text = ""

        # Single-shot Whisper transcription on the full session
        if arr is not None and arr.size > 0:
            audio_float = arr.astype(np.float32) / 32768.0
            try:
                print("🧠 Transcribing full session with Whisper...")
                segments, _ = whisper_model.transcribe(audio_float, language="en")
                parts = []
                for seg in segments:
                    t = seg.text.strip()
                    if t:
                        parts.append(t)
                transcript_text = " ".join(parts).strip()
                if transcript_text:
                    print(f"📝 Transcript: {transcript_text}")
                    text_buffer.set_text(transcript_text + "\n\nNow thinking...")
                    # Optionally show a short preview on the e-paper
                else:
                    text_buffer.set_text("No speech detected.")
            except Exception as e:
                print(f"❌ Transcription error: {e}")
                text_buffer.set_text("Transcription error.")

        # LLM post-processing
        if transcript_text:
            try:
                print("🤖 Sending transcript to GPT-5...")
                answer = get_gpt_5_answer(transcript_text)
                print("🖨️ Printing Markdown answer...")
                print_markdown(answer)
            except Exception as e:
                print(f"❌ Error generating or printing answer: {e}")

        # Cleanup + cooldown + wake model reset
        with state_lock:
            transcription_buffer.clear()
            transcribing = False
            wake_cooldown_until = time.monotonic() + POST_SESSION_COOLDOWN_SEC
            print(f"🛑 Wake detection paused for {POST_SESSION_COOLDOWN_SEC:.1f}s")

        virginize_wake_model()

    def transcription_worker() -> None:
        """Runs while transcribing; exits on max duration or silence timeout."""
        nonlocal silence_start_ts

        print("🎤 Capturing audio...")
        while True:
            now = time.monotonic()
            with state_lock:
                running = transcribing
                started = transcription_start_ts
                silence_started = silence_start_ts

            if not running:
                break

            if now - started >= SESSION_MAX_SEC:
                print("\n⏰ 40-second timeout reached")
                break

            if silence_started > 0 and (now - silence_started) >= SILENCE_TIMEOUT_SEC:
                print("\n🔇 5 seconds of silence detected")
                break

            time.sleep(0.05)

        end_session_and_postprocess()

    # --------------------------
    # Audio callback
    # --------------------------
    def on_audio(indata, frames, time_info, status):
        nonlocal last_trigger_ts, transcribing, transcription_thread
        nonlocal transcription_start_ts, silence_start_ts
        nonlocal consecutive_wake_hits, wake_cooldown_until

        if status:
            # Optionally log status
            pass

        # sounddevice delivers ndarray shaped (frames, channels)
        try:
            audio = indata[:, 0].astype(np.int16, copy=False)
        except Exception:
            audio = np.frombuffer(indata, dtype=np.int16)

        if audio.size < CHUNK_SIZE:
            return

        now = time.monotonic()

        # If currently transcribing, buffer audio + silence tracking
        with state_lock:
            currently_transcribing = transcribing
        if currently_transcribing:
            with state_lock:
                transcription_buffer.extend(audio)
                if is_silence(audio):
                    if silence_start_ts == 0:
                        silence_start_ts = now
                else:
                    silence_start_ts = 0
            return

        # Not transcribing: respect cooldown and retrigger gap
        if now < wake_cooldown_until:
            return
        if (now - last_trigger_ts) < RETRIGGER_GAP_SEC:
            return

        # Wake detection on the first CHUNK_SIZE samples
        scores = oww.predict(audio[:CHUNK_SIZE])
        score = float(scores.get(target_key, 0.0))

        # Hysteresis: require N consecutive frames >= threshold
        if score >= WAKE_SCORE_THRESHOLD:
            consecutive_wake_hits += 1
        else:
            consecutive_wake_hits = 0

        if consecutive_wake_hits >= WAKE_HYSTERESIS_FRAMES:
            # Trigger session
            last_trigger_ts = now
            consecutive_wake_hits = 0
            print(f"\n🔔 Wake word detected! (score={score:.3f})")
            text_buffer.set_text("Listening...")

            with state_lock:
                transcribing = True
                transcription_buffer.clear()
                transcription_start_ts = now
                silence_start_ts = 0.0
                wake_cooldown_until = now + 0.5  # brief guard during transition

            
            transcription_thread = threading.Thread(target=transcription_worker, daemon=True)
            transcription_thread.start()

    # --------------------------
    # Audio stream loop
    # --------------------------
    try:
        with sd.InputStream(
            blocksize=CHUNK_SIZE,
            callback=on_audio,
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype=DTYPE,
        ):
            while True:
                time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nExiting.")
    finally:
        with state_lock:
            running = transcribing
        if running:
            with state_lock:
                transcribing = False
        if transcription_thread and transcription_thread.is_alive():
            transcription_thread.join(timeout=2.0)

if __name__ == "__main__":
    main()