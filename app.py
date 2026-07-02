import os
# CRITICAL: Prevent OpenMP runtime conflicts which cause Access Violation (0xc0000005) in arrow.dll on Windows.
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["ARROW_IO_THREADS"] = "1"
os.environ["ARROW_ENABLE_THREAD_POOL"] = "0"

# Reconfigure stdout/stderr for Indic text (₹ symbol etc.)
import sys
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# CRITICAL: FlagEmbedding must be imported at the absolute top on Windows to prevent DLL load order crashes.
try:
    import FlagEmbedding
except ImportError:
    pass

import tempfile
import time
import numpy as np
import streamlit as st
import torch
from streamlit_webrtc import WebRtcMode, webrtc_streamer
from transformers import pipeline
from huggingface_hub import login
import soundfile as sf
from pathlib import Path

# Optional: SpeechRecognition for simple microphone capture
try:
    import speech_recognition as sr
    sr_enabled = True
except Exception:
    sr = None
    sr_enabled = False

SUPPORTED_LANGUAGES = {
    "hi": "Hindi",
    "bn": "Bengali",
    "ta": "Tamil",
    "te": "Telugu",
    "kn": "Kannada",
    "mr": "Marathi",
    "ml": "Malayalam",
    "gu": "Gujarati",
    "pa": "Punjabi",
}

ASR_MODEL = "openai/whisper-medium"
TRANSLATION_SRC_TO_EN = {
    "hi": "ai4bharat/indic-trans2-hi-en",
    "bn": "ai4bharat/indic-trans2-bn-en",
    "ta": "ai4bharat/indic-trans2-ta-en",
    "te": "ai4bharat/indic-trans2-te-en",
    "kn": "ai4bharat/indic-trans2-kn-en",
    "mr": "ai4bharat/indic-trans2-mr-en",
    "ml": "ai4bharat/indic-trans2-ml-en",
    "gu": "ai4bharat/indic-trans2-gu-en",
    "pa": "ai4bharat/indic-trans2-pa-en",
}
TRANSLATION_EN_TO_SRC = {
    "hi": "ai4bharat/indic-trans2-en-hi",
    "bn": "ai4bharat/indic-trans2-en-bn",
    "ta": "ai4bharat/indic-trans2-en-ta",
    "te": "ai4bharat/indic-trans2-en-te",
    "kn": "ai4bharat/indic-trans2-en-kn",
    "mr": "ai4bharat/indic-trans2-en-mr",
    "ml": "ai4bharat/indic-trans2-en-ml",
    "gu": "ai4bharat/indic-trans2-en-gu",
    "pa": "ai4bharat/indic-trans2-en-pa",
}
TTS_MODEL = "ai4bharat/indic-parler-tts"


@st.cache_resource
def get_asr_pipeline():
    hf_token = st.secrets.get("HUGGINGFACE_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
    if hf_token:
        login(token=hf_token)
    try:
        return pipeline("automatic-speech-recognition", model=ASR_MODEL, chunk_length_s=30, device=0, torch_dtype=torch.float16)
    except OSError as e:
        if "private" in str(e).lower() or "token" in str(e).lower():
            st.warning("Private model requires authentication. Please provide your Hugging Face token.")
            hf_token = st.text_input("Hugging Face Token", type="password", key="hf_token_input")
            if hf_token:
                login(token=hf_token)
                return pipeline("automatic-speech-recognition", model=ASR_MODEL, chunk_length_s=30, device=0, torch_dtype=torch.float16)
        raise


@st.cache_resource
def get_translation_pipeline(model_name: str):
    return pipeline("translation", model=model_name, device=0, torch_dtype=torch.float16)


@st.cache_resource
def get_tts_pipeline():
    return pipeline("text-to-speech", model=TTS_MODEL, device=0, torch_dtype=torch.float16)


def transcribe_audio(audio_path: str) -> str:
    start_time = time.time()
    asr = get_asr_pipeline()
    result = asr(audio_path)
    text = result.get("text", "") if isinstance(result, dict) else str(result)
    print(f"[ASR] Transcription took {time.time() - start_time:.2f} seconds")
    return text


def translate_text(text: str, model_name: str) -> str:
    start_time = time.time()
    translator = get_translation_pipeline(model_name)
    outputs = translator(text, max_length=1024)
    print(f"[Translation] Translation took {time.time() - start_time:.2f} seconds")
    if isinstance(outputs, list) and outputs:
        return outputs[0].get("translation_text", outputs[0].get("text", ""))
    if isinstance(outputs, dict):
        return outputs.get("translation_text", outputs.get("text", ""))
    return str(outputs)


def process_query_with_llm(english_text: str) -> str:
    prompt = english_text.lower()
    if "balance" in prompt or "balance" in english_text:
        return "I can help you check your account balance. Please provide your account number or login details in the bank app to view the latest balance."
    if "loan" in prompt or "interest" in prompt:
        return "I can assist with information on loans and interest rates. What type of loan are you interested in?"
    if "transaction" in prompt or "transfer" in prompt:
        return "I can help you initiate a fund transfer. Please confirm the recipient and amount so I can guide you through the next steps."
    return "I am here to help with your banking questions. Please tell me what you need help with, such as checking balance, viewing transactions, or loan details."


def synthesize_speech(text: str, output_path: str) -> str:
    start_time = time.time()
    tts = get_tts_pipeline()
    result = tts(text)
    print(f"[TTS] Speech synthesis took {time.time() - start_time:.2f} seconds")
    if isinstance(result, dict):
        audio = result.get("audio")
        sampling_rate = result.get("sampling_rate", 22050)
    elif isinstance(result, list) and result:
        audio = result[0].get("audio")
        sampling_rate = result[0].get("sampling_rate", 22050)
    else:
        audio = result
        sampling_rate = 22050

    if isinstance(audio, bytes):
        with open(output_path, "wb") as out_file:
            out_file.write(audio)
        return output_path

    if hasattr(audio, "numpy"):
        audio_array = audio.numpy()
    else:
        audio_array = audio

    sf.write(output_path, audio_array, sampling_rate)
    return output_path


def add_history_entry(action: str, prompt: str, result: str, details: str = ""):
    if "history" not in st.session_state:
        st.session_state.history = []
    st.session_state.history.append({
        "time": int(time.time()),
        "action": action,
        "prompt": prompt,
        "result": result,
        "details": details,
    })


def save_audio_frames_to_wav(frames, output_path: str, sampling_rate: int = 48000) -> None:
    if not frames:
        raise ValueError("No audio frames received from microphone.")

    audio_arrays = []
    for frame in frames:
        audio_data = frame.to_ndarray()
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)
        if len(audio_data.shape) > 1:
            audio_data = audio_data.squeeze()
        audio_arrays.append(audio_data)

    audio = np.concatenate(audio_arrays, axis=0)
    audio = np.clip(audio, -1.0, 1.0)
    try:
        sf.write(output_path, audio, sampling_rate)
    except Exception as e:
        import scipy.io.wavfile as wavfile
        audio_int16 = np.int16(audio * 32767)
        wavfile.write(output_path, sampling_rate, audio_int16)


def capture_microphone_audio(ctx, duration_s: int = 5) -> str | None:
    """Capture audio from microphone for a specified duration."""
    if ctx.audio_receiver is None:
        return None

    frames = []
    frame_count = 0
    max_frames = int(duration_s * 50)

    while frame_count < max_frames:
        try:
            received_frames = ctx.audio_receiver.get_frames(timeout=0.5)
            if received_frames:
                frames.extend(received_frames)
                frame_count += len(received_frames)
        except Exception:
            continue

    if not frames:
        return None

    output_path = os.path.join(tempfile.gettempdir(), f"bankassist_mic_{int(time.time())}.wav")
    try:
        save_audio_frames_to_wav(frames, output_path)
        return output_path
    except Exception as e:
        st.error(f"Failed to save audio: {e}")
        return None


def main():
    st.set_page_config(page_title="BankAssist Voice Assistant", page_icon="💬", layout="centered")
    st.title("BankAssist Multilingual Voice Assistant")
    st.markdown(
        "This prototype accepts Indian language audio, transcribes it, translates it into English, generates a banking response, back-translates the reply, and returns speech output."
    )

    source_language = st.selectbox("Select your language", options=list(SUPPORTED_LANGUAGES.keys()), format_func=lambda x: f"{SUPPORTED_LANGUAGES[x]} ({x})")
    audio_input_method = st.radio(
        "Audio input method",
        ["Upload an existing file", "Record from microphone"],
    )

    # Initialize session_state defaults used by the audio UI
    if "uploaded_audio_name" not in st.session_state:
        st.session_state.uploaded_audio_name = ""
    if "uploaded_audio_size" not in st.session_state:
        st.session_state.uploaded_audio_size = 0
    if "audio_transcription" not in st.session_state:
        st.session_state.audio_transcription = ""
    if "audio_source" not in st.session_state:
        st.session_state.audio_source = ""
    if "audio_bytes" not in st.session_state:
        st.session_state.audio_bytes = None

    audio_path = None

    if audio_input_method == "Upload an existing file":
        uploaded_file = st.file_uploader(
            "Upload an audio file to transcribe",
            type=["wav", "mp3", "m4a", "ogg", "flac", "aac"],
        )
        if uploaded_file is None:
            if st.session_state.uploaded_audio_name:
                st.session_state.uploaded_audio_name = ""
                st.session_state.uploaded_audio_size = 0
                st.session_state.audio_transcription = ""
                st.session_state.audio_source = ""
                st.session_state.audio_bytes = None
                st.info("Audio upload cleared. Upload a new file to transcribe.")
        else:
            if (
                uploaded_file.name != st.session_state.uploaded_audio_name
                or uploaded_file.size != st.session_state.uploaded_audio_size
                or st.session_state.audio_source != "upload"
            ):
                suffix = Path(uploaded_file.name).suffix or ".wav"
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_audio:
                    temp_audio.write(uploaded_file.getbuffer())
                    temp_audio_path = temp_audio.name

                try:
                    transcription = transcribe_audio(temp_audio_path)
                    st.session_state.audio_transcription = transcription
                    st.session_state.uploaded_audio_name = uploaded_file.name
                    st.session_state.uploaded_audio_size = uploaded_file.size
                    st.session_state.audio_source = "upload"
                    st.session_state.audio_bytes = uploaded_file.getbuffer()
                    st.success("Audio transcription complete. You can use the transcription as the prompt below.")
                    add_history_entry(
                        action="transcribe_audio",
                        prompt=f"Uploaded audio file: {uploaded_file.name}",
                        result=transcription,
                        details="Audio file transcribed successfully.",
                    )
                except ValueError as exc:
                    st.session_state.audio_transcription = ""
                    st.warning(str(exc))
                    add_history_entry(
                        action="transcribe_audio",
                        prompt=f"Uploaded audio file: {uploaded_file.name}",
                        result="",
                        details=str(exc),
                    )
                except Exception as exc:
                    st.session_state.audio_transcription = ""
                    st.error(f"Failed to transcribe audio: {exc}")
                    add_history_entry(
                        action="transcribe_audio",
                        prompt=f"Uploaded audio file: {uploaded_file.name}",
                        result="",
                        details=str(exc),
                    )
                finally:
                    if os.path.exists(temp_audio_path):
                        try:
                            os.remove(temp_audio_path)
                        except OSError:
                            pass
    else:
        if not sr_enabled:
            st.warning(
                "Microphone transcription requires the SpeechRecognition package. "
                "Install it with `pip install SpeechRecognition` and reload the app."
            )
        else:
            def clear_mic_audio():
                st.session_state.audio_transcription = ""
                st.session_state.audio_bytes = None
                st.session_state.audio_source = ""
                st.session_state.uploaded_audio_name = ""
                st.session_state.uploaded_audio_size = 0

            if st.button("Start microphone recording", key="start_mic_recording"):
                clear_mic_audio()
                recognizer = sr.Recognizer()
                recognizer.dynamic_energy_threshold = True
                recognizer.energy_threshold = 300
                recognizer.pause_threshold = 0.5
                recognizer.non_speaking_duration = 0.5
                try:
                    st.info("Preparing microphone. Please wait...")
                    with sr.Microphone() as source:
                        recognizer.adjust_for_ambient_noise(source, duration=1)
                        st.info("Recording will begin in 1 second. Speak clearly after the prompt.")
                        time.sleep(1)
                        st.info("Recording now...")
                        audio_data = recognizer.listen(source, timeout=5, phrase_time_limit=15)
                    transcription = recognizer.recognize_google(audio_data)
                    st.session_state.audio_transcription = transcription
                    st.session_state.audio_source = "microphone"
                    st.session_state.audio_bytes = audio_data.get_wav_data()
                    st.success("Microphone recording transcribed successfully.")
                    add_history_entry(
                        action="transcribe_audio",
                        prompt="Recorded from microphone",
                        result=transcription,
                        details="Microphone audio transcribed successfully.",
                    )
                except sr.WaitTimeoutError:
                    st.warning("No microphone input detected within the timeout period.")
                    add_history_entry(
                        action="transcribe_audio",
                        prompt="Recorded from microphone",
                        result="",
                        details="No audio detected within timeout.",
                    )
                except sr.UnknownValueError:
                    st.warning("Could not understand the microphone audio. Please try again.")
                    clear_mic_audio()
                    add_history_entry(
                        action="transcribe_audio",
                        prompt="Recorded from microphone",
                        result="",
                        details="Speech not understood.",
                    )
                except Exception as exc:
                    st.error(f"Failed to transcribe microphone audio: {exc}")
                    clear_mic_audio()
                    add_history_entry(
                        action="transcribe_audio",
                        prompt="Recorded from microphone",
                        result="",
                        details=str(exc),
                    )

            if st.session_state.audio_source == "microphone" and st.session_state.audio_bytes:
                st.write("### Microphone recording playback")
                st.audio(st.session_state.audio_bytes, format="audio/wav")


    # Determine transcript source: prefer session_state transcription (mic or upload),
    # otherwise fall back to ASR on an uploaded file path.
    transcript = ""
    if st.session_state.get("audio_transcription"):
        transcript = st.session_state.audio_transcription
        st.subheader("Step 1: ASR Transcript")
        st.text_area("Transcript", transcript, height=120)
    elif audio_path is not None:
        st.subheader("Step 1: ASR Transcript")
        transcript = transcribe_audio(audio_path)
        st.text_area("Transcript", transcript, height=120)
    else:
        transcript = ""

    if transcript:
        st.subheader("Step 2: Translate to English")
        src_to_en_model = TRANSLATION_SRC_TO_EN.get(source_language)
        if src_to_en_model is None:
            st.error(f"Translation model not configured for {source_language}.")
            return
        english_query = translate_text(transcript, src_to_en_model)
        st.text_area("English Query", english_query, height=120)

        st.subheader("Step 3: Process Query")
        english_response = process_query_with_llm(english_query)
        st.text_area("English Response", english_response, height=120)

        st.subheader("Step 4: Translate Response Back")
        en_to_src_model = TRANSLATION_EN_TO_SRC.get(source_language)
        if en_to_src_model is None:
            st.error(f"Back-translation model not configured for {source_language}.")
            return
        back_translated = translate_text(english_response, en_to_src_model)
        st.text_area("Response in Your Language", back_translated, height=120)

        st.subheader("Step 5: Generate Speech")
        audio_path = os.path.join(tempfile.gettempdir(), f"bankassist_response_{source_language}.wav")
        try:
            synthesize_speech(back_translated, audio_path)
            with open(audio_path, "rb") as audio_file:
                audio_bytes = audio_file.read()
            st.audio(audio_bytes, format="audio/wav")
            st.success("Speech output generated successfully.")
        except Exception as error:
            st.error(f"TTS generation failed: {error}")

        st.caption("Note: This prototype uses model-based offline components and a placeholder LLM response engine. Qwen integration can be added later.")

    else:
        st.info("Upload or record audio to begin the pipeline.")


if __name__ == "__main__":
    main()
