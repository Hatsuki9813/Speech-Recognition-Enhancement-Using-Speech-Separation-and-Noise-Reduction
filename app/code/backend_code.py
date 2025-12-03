import subprocess
import os
import librosa
import numpy as np
import ffmpeg
import shutil
import tempfile
import argparse
import torch
from pathlib import Path
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import string
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
def normalize_text(text):
    if text is None:
        return ""
    text = text.lower()  # Make everything lowercase
    text = text.translate(str.maketrans('', '', string.punctuation)) # Remove symbols like , . ! ?
    text = " ".join(text.split()) # Remove extra spaces
    return text
def run_mossformer_inference(input_path, output_dir):
    network = "MossFormer2_SS_16K"
    config = f"../../speech_separation/config/inference/{network}.yaml"
    
    

    cmd = [
        "python3", "-u", "../../speech_separation/inference.py",
        "--config", config,
        "--checkpoint-dir", f"../../speech_separation/checkpoints/{network}",
        "--network", network,
        "--input-path", input_path,
        "--output-dir", output_dir,
    ]

    # Chạy và stream log realtime ra terminal
    result = subprocess.run(cmd, capture_output=True, text=True)
    # In log realtime
    
    return result.stdout

def run_whisper_inference(file_path):
    processor = WhisperProcessor.from_pretrained("openai/whisper-large-v2")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v2")

    try:
        # 1. Load the audio file
        audio, sr = librosa.load(file_path, sr=16000, mono=True)
        
        # 2. Prepare audio for the
        inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
        input_features = inputs.input_features
        attention_mask = inputs.get("attention_mask")

        # 3. Generates text (prediction)
        with torch.no_grad():
            if attention_mask is not None:
                predicted_ids = model.generate(input_features, attention_mask=attention_mask)
            else:
                predicted_ids = model.generate(input_features)

        # 4. Decode computer numbers back to human text
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        return normalize_text(transcription)
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return "" 
def preprocessing_wav_file(input_path):
    try:
        if not os.path.exists(input_path):
            print(f"[ERROR] File không tồn tại: {input_path}")
            return False

        model = os.path.join(BASE_DIR, "..", "model", "mp.rnnn")
        model = os.path.abspath(model)

        if not os.path.exists(model):
            print(f"[ERROR] Model không tồn tại: {model}")
            return False

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp_output = tmp.name

        # Chạy ffmpeg
        try:
            (
                ffmpeg
                .input(input_path)
                .output(
                    tmp_output,
                    af=f"arnndn=m={model}",
                    ac=1,
                    ar=16000,
                    format='wav'
                )
                .overwrite_output()
                .run(quiet=True)
            )
        except ffmpeg.Error as e:
            print("[ERROR] FFmpeg lỗi:", e)
            return False

        # Ghi đè file gốc bằng file đã xử lý
        shutil.move(tmp_output, input_path)

        print("[OK] Preprocessing thành công:", input_path)
        return True

    except Exception as e:
        print("[EXCEPTION] Lỗi không xác định:", e)
        return False



        