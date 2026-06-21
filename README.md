# BankAssist Multilingual Voice Assistant

A Streamlit-based prototype for a banking voice assistant that supports Indian languages through ASR, translation, and TTS. This first version implements:

- IndicWhisper-based speech-to-text for Indian language audio
- IndicTrans2 translation from Indian languages to English and back
- A placeholder English banking response engine
- IndicParler-TTS speech generation for the final reply

## Setup

1. Create and activate a Python environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Run the Streamlit app:

```powershell
streamlit run app.py
```

## Usage

1. Open the local Streamlit URL shown in your terminal.
2. Select the source language.
3. Upload a WAV file containing spoken Indian language audio.
4. Review the transcript, English translation, generated response, and back-translated reply.
5. Play the generated WAV output.

## Notes

- The current implementation uses a simple placeholder LLM function for banking responses.
- Qwen integration is intentionally deferred to a later phase.
- If you want a microphone recording UX, that can be added in the next iteration.
