from huggingface_hub import InferenceClient

client = InferenceClient(
    provider="hf-inference",
    api_key="YOUR_HF_TOKEN"
)

with open("06-20-2026 19.48(2) (2).m4a", "rb") as f:
    result = client.automatic_speech_recognition(f)

print(result)