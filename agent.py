import sounddevice as sd
import soundfile as sf
import os
import threading
import numpy as np
import time
import requests
from pynput import keyboard
from mlx_lm import load, generate

from stream_audio import stream_generated_audio

API_URL = "http://localhost:50250"
TTS_URL = "http://localhost:50252"

# Microphone setup
mic_index = sd.default.device[0]  # default input device
mic_sr = int(sd.query_devices(mic_index)["default_samplerate"])
lock = threading.Lock()

# Mutable globals
audio_buffer = []
recording = False
stream = None
start_delay = 0.2  # seconds to wait after keypress before recording starts

conversation_history = []

# Keep track of held keys to prevent OS key-repeat from triggering multiple streams
held_keys = set()

# load LLM
model, tokenizer = load("mlx-community/LFM2.5-1.2B-Instruct-8bit")

# def normalize(audio, target_rms=0.15):
#     rms = np.sqrt(np.mean(audio**2))
#     if rms > 0:
#         audio = audio * (target_rms / rms)
#     return np.clip(audio, -1.0, 1.0)


def transcribe_audio(fpath: str, mode: str):
    files = {
        "file": open(fpath, "rb")
    }
    data = {
        "temperature": "0.0",
        "temperature_inc": "0.2",
        "response_format": "json",
        "language": "en" if mode == "en2fr" else "fr",
    }
    response = requests.post(f"{API_URL}/inference", files=files, data=data)
    response.raise_for_status()
    text = response.json()["text"]
    # TODO normalise?
    return text


def generate_response(spoken_text: str, mode: str):
    data = (
        {
            "messages": [
                {
                    "role": "user",
                    "content": f'Respond with ONLY the following translated into French:\n\n"{spoken_text}"',
                },
            ]
        }
        if mode == "en2fr"
        else {
            "messages": [
                {
                    "role": "user",
                    "content": f'Respond with ONLY the following translated into English:\n\n"{spoken_text}"',
                },
            ]
        }
    )
    print(f"DEBUG: Sending to LLM:\n{data['messages'][0]['content']}\n")
    prompt = tokenizer.apply_chat_template(data["messages"], add_generation_prompt=True)
    response = generate(model, tokenizer, prompt)
    text = response.replace('"', "").replace("»", "").replace("«", "").strip()
    if mode == "fr2en":
        text = text.replace("Eli", "eelye").replace(
            "eli", "eelye"
        )  # name pronunciation hack
    return text

def audio_callback(indata, frames, time_info, status):
    if recording:
        with lock:
            audio_buffer.append(indata.copy())


def start_recording():
    global recording, audio_buffer, stream
    
    with lock:
        if recording:
            return  # Prevent multiple streams from starting
        print("Recording…")
        recording = True
        audio_buffer = []

    time.sleep(start_delay)

    stream = sd.InputStream(
        samplerate=mic_sr,
        channels=1,
        dtype='int16',
        device=mic_index,
        callback=audio_callback
    )
    stream.start()


def stop_recording(mode: str):
    global recording, audio_buffer, stream
    with lock:
        if recording:
            recording = False
            print("Stopped.")

    if stream is not None:
        stream.stop()
        stream.close()

    if len(audio_buffer) > 0:
        print("Processing audio...")
        audio = np.concatenate(audio_buffer, axis=0)
        # --- Hack: remove the button press noises ---
        trim_duration = 0.05  # seconds to remove from start and end
        trim_samples = int(trim_duration * mic_sr)

        # Ensure the recording is long enough to survive the trim
        if len(audio) > (2 * trim_samples):
            audio = audio[trim_samples : -trim_samples]
        else:
            print("Warning: Recording was too short to trim without deleting the whole file.")
        # audio = normalize(audio)
        sf.write("temp.wav", audio, mic_sr)

        print("Transcribing audio...")
        transcription = transcribe_audio("temp.wav", mode).strip()
        print(f"USER INPUT: '{transcription}'")

        print(f"Generating response in {mode} mode...")
        response = generate_response(transcription, mode)
        normalised = response
        print(f"NORMALISED RESPONSE: '{normalised}'")

        print("Generating audio...")
        ref_codes_path = "samples/juliette.pt" if mode == "en2fr" else "samples/jo.pt"
        ref_text = "samples/juliette.txt" if mode == "en2fr" else "samples/jo.txt"
        target_lang = "french" if mode == "en2fr" else "english"
        stream_generated_audio(
            normalised,
            ref_codes_path=ref_codes_path,
            ref_text=ref_text,
            language=target_lang,
        )


# --- Keyboard Event Listeners for Toggle (macOS) ---

# Track which mode we are currently recording in
current_mode = None  

def on_press(key):
    global recording, current_mode

    try:
        # --- CHAT TOGGLE ('z') ---
        if key.char == 'z':
            if not recording:
                current_mode = "en2fr"
                print("\n[STARTED] Recording en2fr mode. Press 'z' again to stop.")
                # Start in a background thread so time.sleep doesn't freeze the listener
                threading.Thread(target=start_recording).start()

            elif current_mode == "en2fr":
                print("\n[STOPPED] en2fr recording finished. Processing...")
                # Run the heavy processing (LLM/TTS) in a background thread
                threading.Thread(target=stop_recording, args=("en2fr",)).start()
                current_mode = None

            elif current_mode == "fr2en":
                print("Currently in fr2en mode! Press 'x' to stop it first.")

        # --- TRANSLATE TOGGLE ('x') ---
        elif key.char == 'x':
            if not recording:
                current_mode = "fr2en"
                print("\n[STARTED] Recording fr2en mode. Press 'x' again to stop.")
                threading.Thread(target=start_recording).start()

            elif current_mode == "fr2en":
                print("\n[STOPPED] fr2en recording finished. Processing...")
                threading.Thread(target=stop_recording, args=("fr2en",)).start()
                current_mode = None

            elif current_mode == "en2fr":
                print("Currently in en2fr mode! Press 'z' to stop it first.")

    except AttributeError:
        # Ignore special keys (Shift, Ctrl, etc.)
        pass

    # Allow exiting the script cleanly using the ESC key
    if key == keyboard.Key.esc:
        print("Exiting...")
        return False

print("Toggle-to-talk ready.")
print("  - Press 'z' to start/stop en2fr.")
print("  - Press 'x' to start/stop fr2en.")
print("  - Press ESC to quit.")

# We only need on_press now; no need for on_release
with keyboard.Listener(on_press=on_press) as listener:
    listener.join()
