import os
import pyaudio
import requests
from fire import Fire

API_URL = "http://localhost:50252"


def stream_generated_audio(
    input_text,
    ref_codes_path="samples/juliette.pt",
    ref_text="samples/juliette.txt",
    language="french",
):

    # Check if ref_text is a path if it is read it if not just return string
    if ref_text and os.path.exists(ref_text):
        with open(ref_text, "r") as f:
            ref_text = f.read().strip()

    # prepare request data
    data = {
        "text": input_text,
        "ref_codes_path": ref_codes_path,
        "ref_text": ref_text,
        "language": language,
    }

    # set up stream
    print(f"Generating audio for input text: {input_text}")
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=24_000, output=True)

    # stream
    print("Streaming...")
    response = requests.post(f"{API_URL}/generate-streaming", json=data, stream=True)
    for chunk in response.iter_content(chunk_size=None):
        stream.write(chunk)

    stream.stop_stream()
    stream.close()
    p.terminate()


if __name__ == "__main__":
    Fire(stream_generated_audio)
