"""
indic_tts.py
English → Indian Language TTS
Translates English text to a chosen Indian language, then generates
audio using ai4bharat/indic-parler-tts.

Usage:
    python indic_tts_final.py --text "Hello, how are you?" --lang hi --out output.mp3

Requirements:
    pip install parler-tts transformers torch pydub deep-translator nltk numpy
"""

import io
import argparse
import numpy as np
import torch
import nltk
# Use stdlib wave + ffmpeg to avoid depending on pydub/audioop on Windows
from deep_translator import GoogleTranslator
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer, AutoFeatureExtractor

nltk.download("punkt_tab", quiet=True)

# ── Language config ────────────────────────────────────────────────────────────
# Maps: user-facing lang code → (Google Translate code, speaker name, script hint)
LANGUAGES = {
    "hi": ("hi",  "Rohit",   "Hindi"),
    "mr": ("mr",  "Sunita",  "Marathi"),
    "bn": ("bn",  "Aditi",   "Bengali"),
    "te": ("te",  "Prakash", "Telugu"),
    "ta": ("ta",  "Jaya",    "Tamil"),
    "kn": ("kn",  "Suresh",  "Kannada"),
    "ml": ("ml",  "Anjali",  "Malayalam"),
    "gu": ("gu",  "Divya",   "Gujarati"),
    "pa": ("pa",  "Gurpreet","Punjabi"),
    "or": ("or",  "Priya",   "Odia"),
    "as": ("as",  "Mitali",  "Assamese"),
    "ur": ("ur",  "Zara",    "Urdu"),
    "sa": ("sa",  "Arjun",   "Sanskrit"),
    "mai":("mai", "Sunita",  "Maithili"),   # limited GT support
    "ne": ("ne",  "Sita",    "Nepali"),
    "sd": ("sd",  "Rashida", "Sindhi"),
    "kok":("kok", "Kavita",  "Konkani"),    # limited GT support
    "doi":("doi", "Anita",   "Dogri"),      # limited GT support
    "mni":("mni", "Ibemhal", "Manipuri"),  # limited GT support
    "ks": ("ks",  "Rafia",   "Kashmiri"),   # limited GT support
    "bho":("bho", "Ramu",    "Bhojpuri"),   # limited GT support
    "sat":("sat", "Sita",    "Santali"),    # limited GT support
}

DESCRIPTION_TEMPLATE = (
    "{speaker} speaks at a moderate pace with a clear, neutral tone. "
    "The recording is very high quality with no background noise."
)

# ── Device & model setup ───────────────────────────────────────────────────────
device = (
    "cuda:0" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
torch_dtype = torch.bfloat16 if device != "cpu" else torch.float32

FINETUNED_REPO = "ai4bharat/indic-parler-tts"

print(f"Loading model on {device} …")
model = ParlerTTSForConditionalGeneration.from_pretrained(
    FINETUNED_REPO, attn_implementation="eager", torch_dtype=torch_dtype
).to(device)

tokenizer            = AutoTokenizer.from_pretrained(FINETUNED_REPO)
description_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
feature_extractor    = AutoFeatureExtractor.from_pretrained(FINETUNED_REPO)

sampling_rate = model.audio_encoder.config.sampling_rate


# ── Helpers ────────────────────────────────────────────────────────────────────
def translate(text: str, target_lang_code: str) -> str:
    """Translate English text → target Indian language via Google Translate."""
    if target_lang_code == "en":
        return text
    try:
        translated = GoogleTranslator(source="en", target=target_lang_code).translate(text)
        return translated
    except Exception as e:
        print(f"[warn] Translation failed ({e}); using original English text.")
        return text


def numpy_to_mp3(audio_array: np.ndarray, sr: int) -> bytes:
    import wave
    import tempfile
    import subprocess
    import os

    if np.issubdtype(audio_array.dtype, np.floating):
        max_val = np.max(np.abs(audio_array))
        if max_val > 0:
            audio_array = (audio_array / max_val) * 32767
        audio_array = audio_array.astype(np.int16)

    # write temporary WAV using stdlib (mono, PCM16)
    tmp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp_wav.close()
    try:
        with wave.open(tmp_wav.name, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(audio_array.dtype.itemsize)
            wf.setframerate(sr)
            wf.writeframes(audio_array.tobytes())

        # convert WAV -> MP3 using ffmpeg (must be in PATH)
        tmp_mp3 = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
        tmp_mp3.close()
        cmd = [
            "ffmpeg", "-loglevel", "error", "-y",
            "-i", tmp_wav.name,
            "-b:a", "320k",
            tmp_mp3.name,
        ]
        subprocess.run(cmd, check=True)

        with open(tmp_mp3.name, "rb") as f:
            data = f.read()
    finally:
        try:
            os.unlink(tmp_wav.name)
        except Exception:
            pass
        try:
            os.unlink(tmp_mp3.name)
        except Exception:
            pass

    return data


def chunk_text(text: str, chunk_size: int = 25) -> list[str]:
    """Split text into sentence-aware chunks of ≤ chunk_size words."""
    sentences = nltk.sent_tokenize(text)
    chunks, current = [], ""
    for sent in sentences:
        candidate = f"{current} {sent}".strip()
        if len(candidate.split()) >= chunk_size:
            if current:
                chunks.append(current)
            current = sent
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks or [text]


def generate_audio(translated_text: str, description: str) -> np.ndarray:
    """Run TTS on translated_text, return raw numpy audio."""
    desc_inputs = description_tokenizer(description, return_tensors="pt").to(device)
    chunks = chunk_text(translated_text)
    print(f"  Chunks: {chunks}")

    all_audio = []
    for chunk in chunks:
        prompt = tokenizer(chunk, return_tensors="pt").to(device)
        generation = model.generate(
            input_ids=desc_inputs.input_ids,
            attention_mask=desc_inputs.attention_mask,
            prompt_input_ids=prompt.input_ids,
            prompt_attention_mask=prompt.attention_mask,
            do_sample=True,
            return_dict_in_generate=True,
        )
        if hasattr(generation, "sequences") and hasattr(generation, "audios_length"):
            audio = generation.sequences[0, : generation.audios_length[0]]
            audio_np = audio.to(torch.float32).cpu().numpy().squeeze()
            if audio_np.ndim > 1:
                audio_np = audio_np.flatten()
            all_audio.append(audio_np)

    return np.concatenate(all_audio) if all_audio else np.zeros(1)


# ── Main entry ─────────────────────────────────────────────────────────────────
def tts(
    english_text: str,
    lang: str,
    out_path: str = "output.mp3",
    description: str | None = None,
) -> str:
    """
    Translate english_text → lang and synthesise speech.
    Returns the path to the saved .mp3 file.
    """
    if lang not in LANGUAGES:
        raise ValueError(
            f"Unknown language '{lang}'. Available: {', '.join(LANGUAGES)}"
        )

    gt_code, speaker, lang_name = LANGUAGES[lang]
    desc = description or DESCRIPTION_TEMPLATE.format(speaker=speaker)

    print(f"\n[1/3] Translating to {lang_name} …")
    translated = translate(english_text, gt_code)
    print(f"  → {translated}")

    print(f"[2/3] Generating audio with speaker '{speaker}' …")
    audio_np = generate_audio(translated, desc)

    print(f"[3/3] Saving to {out_path} …")
    mp3_bytes = numpy_to_mp3(audio_np, sampling_rate)
    with open(out_path, "wb") as f:
        f.write(mp3_bytes)

    duration = round(audio_np.shape[0] / sampling_rate, 2)
    print(f"Done! {duration}s audio saved to '{out_path}'")
    return out_path


# ── CLI ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="English → Indian Language TTS using Indic Parler-TTS"
    )
    parser.add_argument("--text",  required=True, help="English source text")
    parser.add_argument(
        "--lang", required=True,
        choices=list(LANGUAGES.keys()),
        help=f"Target language code. Options: {', '.join(LANGUAGES.keys())}"
    )
    parser.add_argument("--out",   default="output.mp3", help="Output MP3 path")
    parser.add_argument(
        "--description", default=None,
        help="Optional TTS speaker description override"
    )
    args = parser.parse_args()

    tts(args.text, args.lang, args.out, args.description)
    