#pip install transformers torch
import torch
from transformers import pipeline
whisper = pipeline("automatic-speech-recognition", "openai/whisper-larrge-v3",torch_dtype=torch.float16, device="cuda:0")
transcription = whisper(r"C:\Users\hp\Desktop\RAG\output_material.mp3")
print(transcription["text"])