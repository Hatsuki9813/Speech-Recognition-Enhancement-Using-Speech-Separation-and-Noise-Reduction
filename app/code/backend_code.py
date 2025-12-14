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
import soundfile as sf
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
    device  = "cpu"
    processor = WhisperProcessor.from_pretrained("openai/whisper-large-v2")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v2").to(device)
    model.eval()

    try:
        # 1. Load the audio file
        audio, sr = librosa.load(file_path, sr=16000, mono=True)
        
        # 2. Prepare audio for the
        inputs = processor(audio, sampling_rate=16000, return_tensors="pt").to(device)
        input_features = inputs.input_features
        attention_mask = inputs.get("attention_mask")

        # 3. Generates text (prediction)
        with torch.no_grad():
            if attention_mask is not None:
                predicted_ids = model.generate(input_features, attention_mask=attention_mask, task="transcribe", num_beams=3,
            temperature=0.0)
            else:
                predicted_ids = model.generate(input_features, task="transcribe", num_beams=3,
            temperature=0.0)

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
                    af=f"arnndn=m={model},volume=1.5,silenceremove=1:0:-50dB",
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

def remove_silence_overwrite(input_path):
    try:
        if not os.path.exists(input_path):
            print(f"[ERROR] File không tồn tại: {input_path}")
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
                    af=(
                        "silenceremove="
                        "stop_periods=-1:stop_threshold=-40dB:stop_silence=0.3"
                        #"stop_periods=-1:stop_duration=1:stop_threshold=-45dB"
                       
                    ),
                    ac=1,
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
        print("[OK] Remove silence thành công:", input_path)
        return True
    except Exception as e:
        print("[EXCEPTION] Lỗi không xác định:", e)
        return False


        