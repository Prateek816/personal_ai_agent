import pyaudio
import websocket
import json
import threading
import time
import wave
from urllib.parse import urlencode
from datetime import datetime
import os
from dotenv import load_dotenv

from langchain_groq import ChatGroq

from elevenlabs.client import ElevenLabs
from elevenlabs.play import play

load_dotenv()

# --- Configuration ---
ASSEMBLYAI_API_KEY = os.getenv("ASSEMBLYAI_API_KEY")
if not ASSEMBLYAI_API_KEY:
    print("Error: ASSEMBLYAI_API_KEY not found in environment variables.")
    exit(1)
llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="openai/gpt-oss-120b",
)
elevenlabs = ElevenLabs(
  api_key=os.getenv("ELEVENLABS_API_KEY"),
)



CONNECTION_PARAMS = {
    "speech_model": "u3-rt-pro",
    "sample_rate": 16000,
}
API_ENDPOINT_BASE_URL = "wss://streaming.assemblyai.com/v3/ws"
API_ENDPOINT = f"{API_ENDPOINT_BASE_URL}?{urlencode(CONNECTION_PARAMS)}"

# Audio Configuration
FRAMES_PER_BUFFER = 800
SAMPLE_RATE = CONNECTION_PARAMS["sample_rate"]
CHANNELS = 1
FORMAT = pyaudio.paInt16

# Global variables
audio = None
stream = None
ws_app = None
audio_thread = None
stop_event = threading.Event()

# WAV recording
recorded_frames = []
recording_lock = threading.Lock()

# --- Transcript State ---
transcript_lock = threading.Lock()

session_finals = []
current_partial = ""
last_speech_time = time.time()
pause_fired = False
PAUSE_THRESHOLD = 3

def AI_response(user_statement):
    llm_response = llm.invoke(user_statement)
    audio = elevenlabs.text_to_speech.convert(
    text= llm_response.content,
    voice_id="JBFqnCBsd6RMkjVDRZzb",  # "George" - browse voices at elevenlabs.io/app/voice-library
    model_id="eleven_v3",
    output_format="mp3_44100_128",
    )

    play(audio)

    return llm_response.content


def get_full_statement():
    with transcript_lock:
        parts = session_finals[:]
        if current_partial.strip():
            parts.append(current_partial.strip())
        return " ".join(parts).strip()


last_user_statement = ""

def on_pause_detected():
    global last_user_statement
    
    statement = get_full_statement()
    if not statement:
        return

    last_user_statement = statement
    print(f"\n[USER SAID]: {statement}")
    print(f"[LLM]: {AI_response(last_user_statement)}\n")
    with transcript_lock:
        global session_finals, current_partial
        session_finals = []
        current_partial = ""

def pause_monitor():
    global pause_fired

    while not stop_event.is_set():
        time.sleep(0.2)

        with transcript_lock:
            has_content = bool(session_finals or current_partial.strip())

        if not has_content:
            pause_fired = False
            continue

        elapsed = time.time() - last_speech_time

        if elapsed >= PAUSE_THRESHOLD:
            if not pause_fired:
                pause_fired = True
                on_pause_detected()
        else:
            pause_fired = False


def save_wav_file():
    if not recorded_frames:
        print("No audio data recorded.")
        return
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"recorded_audio_{timestamp}.wav"


def on_open(ws):
    print("WebSocket connection opened.")

    def stream_audio():
        global stream
        print("Starting audio streaming...")
        while not stop_event.is_set():
            audio_data = stream.read(FRAMES_PER_BUFFER, exception_on_overflow=False)
            with recording_lock:
                recorded_frames.append(audio_data)
            ws.send(audio_data, websocket.ABNF.OPCODE_BINARY)
        print("Audio streaming stopped.")

    global audio_thread
    audio_thread = threading.Thread(target=stream_audio)
    audio_thread.daemon = True
    audio_thread.start()


def on_message(ws, message):
    global last_speech_time, current_partial

    data = json.loads(message)
    msg_type = data.get('type')

    if msg_type == "Begin":
        session_id = data.get('id')
        expires_at = data.get('expires_at')
        print(f"\nSession began: ID={session_id}, ExpiresAt={datetime.fromtimestamp(expires_at)}")

    elif msg_type == "Turn":
        transcript = data.get('transcript', '').strip()

        if not transcript:
            return

        last_speech_time = time.time()

        with transcript_lock:
            if data.get('end_of_turn'):
                session_finals.append(transcript)
                current_partial = ""
                full = " ".join(session_finals)
                print(f"\r[Captured] {full}" + " " * 20)
            else:
                current_partial = transcript
                full = " ".join(session_finals)
                preview = (full + " " + transcript).strip()
                print(f"\r[Live]     {preview}", end='', flush=True)

    elif msg_type == "Termination":
        audio_duration = data.get('audio_duration_seconds', 0)
        session_duration = data.get('session_duration_seconds', 0)
        print(f"\nSession Terminated: Audio={audio_duration}s, Session={session_duration}s")


def on_error(ws, error):
    print(f"\nWebSocket Error: {error}")
    stop_event.set()


def on_close(ws, close_status_code, close_msg):
    print(f"\nWebSocket Disconnected: Status={close_status_code}, Msg={close_msg}")
    save_wav_file()
    global stream, audio
    stop_event.set()
    if stream:
        if stream.is_active():
            stream.stop_stream()
        stream.close()
        stream = None
    if audio:
        audio.terminate()
        audio = None
    if audio_thread and audio_thread.is_alive():
        audio_thread.join(timeout=1.0)


def run():
    global audio, stream, ws_app

    audio = pyaudio.PyAudio()

    stream = audio.open(
        input=True,
        frames_per_buffer=FRAMES_PER_BUFFER,
        channels=CHANNELS,
        format=FORMAT,
        rate=SAMPLE_RATE,
    )
    print("Microphone stream opened.")
    print(f"Speak naturally. After {PAUSE_THRESHOLD}s of silence, your full statement is captured.")
    print("Press Ctrl+C to stop.\n")

    ws_app = websocket.WebSocketApp(
        API_ENDPOINT,
        header={"Authorization": ASSEMBLYAI_API_KEY},
        on_open=on_open,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close,
    )

    monitor_thread = threading.Thread(target=pause_monitor)
    monitor_thread.daemon = True
    monitor_thread.start()

    ws_thread = threading.Thread(target=ws_app.run_forever)
    ws_thread.daemon = True
    ws_thread.start()

    while ws_thread.is_alive():
        time.sleep(0.1)

    if ws_app and ws_app.sock and ws_app.sock.connected:
        ws_app.send(json.dumps({"type": "Terminate"}))
        time.sleep(3)

    if ws_app:
        ws_app.close()

    ws_thread.join(timeout=2.0)

    if stream and stream.is_active():
        stream.stop_stream()
    if stream:
        stream.close()
    if audio:
        audio.terminate()
    print("Cleanup complete. Exiting.")


if __name__ == "__main__":
    run()
    print(last_user_statement)