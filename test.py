#!/usr/bin/env python3
"""
Wakeword listener for: "hey jarvis" using OpenWakeWord.

Usage:
  python hey_jarvis_listener.py                # load default bundled models
  python hey_jarvis_listener.py --model-path /path/to/hey_jarvis_v0.1.tflite
  python hey_jarvis_listener.py --threshold 0.5 --debounce 1.5
"""

import argparse
import time
from collections import deque

import numpy as np
import sounddevice as sd
from openwakeword.model import Model

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", type=str, default="",
                   help="Path to a specific wakeword model file (e.g., hey_jarvis_v0.1.tflite).")
    p.add_argument("--threshold", type=float, default=0.5,
                   help="Activation threshold (higher = fewer false positives).")
    p.add_argument("--debounce", type=float, default=1.5,
                   help="Seconds to wait after a detection before allowing another trigger.")
    p.add_argument("--rate", type=int, default=16000, help="Sample rate; OpenWakeWord expects 16 kHz.")
    p.add_argument("--chunk", type=int, default=1280,
                   help="Samples per chunk (1280 ~= 80 ms at 16 kHz).")
    args = p.parse_args()

    # Load model(s)
    if args.model_path:
        oww = Model(wakeword_models=[args.model_path])
        print(f"Loaded model from: {args.model_path}")
        target_key = args.model_path.split("/")[-1].replace(".tflite", "").replace(".onnx", "")
    else:
        # Loads the default set of bundled models (if present)
        oww = Model()
        # Try to pick the "hey_jarvis" model key from whatever is bundled/available.
        # Fall back to the first model if not found, but warn.
        keys = list(oww.models.keys())
        target_key = None
        for k in keys:
            if "hey_jarvis" in k.lower() or "jarvis" in k.lower():
                target_key = k
                break
        if target_key is None:
            target_key = keys[0] if keys else None
            print("⚠️ Couldn’t find a bundled 'hey_jarvis' model. "
                  "Pass --model-path to a hey_jarvis .tflite/.onnx file.")
    if target_key is None:
        raise RuntimeError("No wakeword models are loaded; cannot proceed.")

    print(f"Listening for wakeword using model key: {target_key}")
    print("Press Ctrl+C to stop.\n")

    # Audio stream config (mono 16 kHz)
    sd.default.channels = 1
    sd.default.samplerate = args.rate
    sd.default.dtype = ("int16", "int16")

    last_trigger = 0.0
    recent_scores = deque(maxlen=30)  # for a tiny bit of smoothing / debugging

    def on_audio(indata, frames, time_info, status):
        nonlocal last_trigger
        if status:
            # sounddevice may report underflows/overflows; not fatal.
            pass

        audio_i16 = np.frombuffer(indata, dtype=np.int16)
        # Ensure exactly CHUNK samples (sounddevice gives us blocksize chunks already)
        if len(audio_i16) < args.chunk:
            return

        # Predict; Model.predict accepts raw int16 at 16 kHz.
        scores = oww.predict(audio_i16[:args.chunk])

        score = float(scores.get(target_key, 0.0))
        recent_scores.append(score)

        now = time.monotonic()
        if score >= args.threshold and (now - last_trigger) >= args.debounce:
            last_trigger = now
            print(f"[{time.strftime('%H:%M:%S')}] 🔔 Wakeword detected! (score={score:.3f})")
            # TODO: call your assistant/handler here

    # Start stream
    with sd.InputStream(blocksize=args.chunk, callback=on_audio):
        try:
            while True:
                # Print a lightweight live view of the current score every ~0.25s
                if recent_scores:
                    print(f"score≈{recent_scores[-1]:.3f}      ", end="\r", flush=True)
                time.sleep(0.25)
        except KeyboardInterrupt:
            print("\nExiting.")

if __name__ == "__main__":
    main()