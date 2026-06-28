import sys
import tempfile
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()
import transformers
import os
from transformers import GenerationConfig, WhisperConfig, WhisperForConditionalGeneration, WhisperProcessor, pipeline
from deep_translator import GoogleTranslator
INDIC_WHISPER_LARGE_DIMS = {
    "d_model": 1280,
    "encoder_layers": 32,
    "decoder_layers": 32,
    "num_hidden_layers": 32,
    "encoder_ffn_dim": 5120,
    "decoder_ffn_dim": 5120,
    "encoder_attention_heads": 20,
    "decoder_attention_heads": 20,
}
LOCAL_CACHE = Path.home() / ".cache" / "huggingface" / "hub" / "models--parthiv11--indic_whisper_nodcil"
MODEL = "parthiv11/indic_whisper_nodcil"

HF_TOKEN = os.environ.get("HUGGINGFACE_TOKEN")
if HF_TOKEN:
    os.environ["HUGGINGFACE_TOKEN"] = HF_TOKEN
else:
    raise ValueError("HUGGINGFACE_TOKEN is not set")

def translate_text(text: str, src_lang: str, tgt_lang: str = "en") -> str:
    """Translate text to English using Google Translate (via deep-translator).

    src_lang should be a Whisper language code (e.g. 'hi', 'mr', 'ta').
    Uses 'auto' detection as fallback if the code isn't recognised.
    """
    translator = GoogleTranslator(source=src_lang, target=tgt_lang)
    return translator.translate(text)

def find_local_model() -> Path | None:
    main_ref = LOCAL_CACHE / "refs" / "main"
    if main_ref.exists():
        snapshot = LOCAL_CACHE / "snapshots" / main_ref.read_text(encoding="utf-8").strip()
        if snapshot.is_dir() and (snapshot / "pytorch_model.bin").exists():
            return snapshot
    for snapshot in sorted(LOCAL_CACHE.glob("snapshots/*"), reverse=True):
        if snapshot.is_dir() and (snapshot / "pytorch_model.bin").exists():
            return snapshot
    return None


def _patch_whisper_generation_config(model: WhisperForConditionalGeneration) -> None:
    """IndicWhisper ships without lang/task mappings in generation_config."""
    if int(transformers.__version__.split(".")[0]) >= 5:
        model.generation_config = GenerationConfig.from_pretrained("openai/whisper-large-v2")


def prepare_audio(audio_path: str) -> tuple[str, bool]:
    ext = Path(audio_path).suffix.lower().lstrip(".")
    if ext in {"wav", "flac", "ogg"}:
        return audio_path, False

    from pydub import AudioSegment

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
        wav_path = temp_audio.name
    AudioSegment.from_file(audio_path).export(wav_path, format="wav")
    return wav_path, True


def whisper_generate_kwargs(processor: WhisperProcessor, language: str, task: str = "transcribe") -> dict:
    """Build generate kwargs compatible with transformers 4.x and 5.x."""
    if int(transformers.__version__.split(".")[0]) >= 5:
        return {"language": language, "task": task}

    forced_decoder_ids = processor.get_decoder_prompt_ids(language=language, task=task)
    if forced_decoder_ids is None:
        raise ValueError(f"Unsupported Whisper language or task: {language}/{task}")
    return {"forced_decoder_ids": forced_decoder_ids}


def load_pipeline(model_source: str | Path):
    config = WhisperConfig.from_pretrained(str(model_source))
    for key, value in INDIC_WHISPER_LARGE_DIMS.items():
        setattr(config, key, value)
    model = WhisperForConditionalGeneration.from_pretrained(str(model_source), config=config)
    _patch_whisper_generation_config(model)
    processor = WhisperProcessor.from_pretrained(str(model_source))
    asr_pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
    )
    return asr_pipe, processor


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python ASRtest.py <audio_file> [language_code]")
        print("Example: python ASRtest.py sample.wav hi")
        sys.exit(1)

    audio_path = sys.argv[1]
    language = sys.argv[2] if len(sys.argv) >= 3 else "hi"
    model_path = find_local_model() or MODEL
    print(f"Loading model from: {model_path}")

    pipe, processor = load_pipeline(model_path)
    asr_path, converted = prepare_audio(audio_path)
    try:
        result = pipe(asr_path, generate_kwargs=whisper_generate_kwargs(processor, language))
    finally:
        if converted:
            Path(asr_path).unlink(missing_ok=True)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    transcribed = result.get("text", result)
    print(transcribed)
    print("Translating to English...")
    english_transcript = translate_text(transcribed, language)   # language == Whisper lang code e.g. 'hi'
    print(english_transcript)
