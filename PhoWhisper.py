import torch
import librosa
import jiwer
import string
from pathlib import Path
from transformers import WhisperProcessor, WhisperForConditionalGeneration

def normalize_text(text):
    if text is None:
        return ""
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = " ".join(text.split())
    return text

def transcribe(model, processor, file_path):
    try:
        audio, sr = librosa.load(file_path, sr=16000, mono=True)
        
        inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
        input_features = inputs.input_features
        attention_mask = inputs.get("attention_mask")

        with torch.no_grad():
            if attention_mask is not None:
                predicted_ids = model.generate(input_features, attention_mask=attention_mask, task="transcribe")
            else:
                predicted_ids = model.generate(input_features, task="transcribe")

        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        return normalize_text(transcription)
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return ""

def main():
    target_dir = Path(r"F:\Whisper\test_case\mix_0007")
    model_id = "vinai/PhoWhisper-large"
    
    print("Loading model on CPU...")
    processor = WhisperProcessor.from_pretrained(model_id)
    model = WhisperForConditionalGeneration.from_pretrained(model_id)
    model.eval()

    mix_name = target_dir.name
    mix_id = mix_name.split('_')[-1]

    files = {
        "txt_s1": target_dir / f"source1_{mix_id}.txt",
        "txt_s2": target_dir / f"source2_{mix_id}.txt",
        "mix_s1": target_dir / f"{mix_name}_s1.wav",
        "mix_s2": target_dir / f"{mix_name}_s2.wav",
        "clean_s1": target_dir / f"source1_{mix_id}.wav",
        "clean_s2": target_dir / f"source2_{mix_id}.wav"
    }

    try:
        with open(files["txt_s1"], 'r', encoding='utf-8') as f: ref_s1 = normalize_text(f.read())
        with open(files["txt_s2"], 'r', encoding='utf-8') as f: ref_s2 = normalize_text(f.read())
    except FileNotFoundError:
        print("Error: Could not find text files.")
        return

    print("Transcribing..")
    hyp_mix_s1 = transcribe(model, processor, files["mix_s1"])
    hyp_mix_s2 = transcribe(model, processor, files["mix_s2"])
    
    hyp_clean_s1 = transcribe(model, processor, files["clean_s1"])
    hyp_clean_s2 = transcribe(model, processor, files["clean_s2"])

    wer_clean_s1 = jiwer.wer(ref_s1, hyp_clean_s1)
    wer_clean_s2 = jiwer.wer(ref_s2, hyp_clean_s2)
    wer_sep_s1 = jiwer.wer(ref_s1, hyp_mix_s1)
    wer_sep_s2 = jiwer.wer(ref_s2, hyp_mix_s2)

    print(f"Results for {mix_name}")
    print(f"WER Clean S1:     {wer_clean_s1:.4f}")
    print(f"WER Clean S2:     {wer_clean_s2:.4f}")
    print(f"WER Separated S1: {wer_sep_s1:.4f}")
    print(f"WER Separated S2: {wer_sep_s2:.4f}")

if __name__ == "__main__":
    main()