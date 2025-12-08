import sys, os
import streamlit as st
import pandas as pd
from io import StringIO
import tempfile
import uuid
from audiorecorder import audiorecorder
from backend_code import preprocessing_wav_file, run_mossformer_inference, run_whisper_inference, normalize_text
from pydub import AudioSegment
UPLOAD_DIR = "../audio"
if "upload_id" not in st.session_state:
    st.session_state.upload_id = str(uuid.uuid4())
if "record_path" not in st.session_state:
    st.session_state.record_path = None
if "saved_record" not in st.session_state:
    st.session_state.saved_record = False
st.set_page_config(
    layout="wide",
    page_title="MOSSFORMER2 - WHISPER SYSTEM",
    page_icon="🎙️"
)

st.markdown(
    """
    <h1 style='text-align:center; color:#1f77b4;'>
        🎧 MOSSFORMER2 – WHISPER SYSTEM
    </h1>
    <h4 style='text-align:center; color:gray;'>
        FFmpeg RNN noise reduction · MossFormer2 speech separation · Whisper Speech-to-Text
    </h4>
    """,
    unsafe_allow_html=True
)
st.write("---")

with st.container(border=True):
    st.subheader("1️⃣ Select Input Method")
    mode = st.radio(
        "Choose how to provide audio:",
        ("Upload file", "Recording"),
        horizontal=True
    )
    if mode == "Upload file":
        uploaded_file = st.file_uploader("Upload .wav", type=["wav"])

        if uploaded_file:
            upload_id = st.session_state.upload_id
            upload_path = os.path.join(UPLOAD_DIR, f"{upload_id}_upload.wav")

            # Save ONCE only
            if "saved_upload" not in st.session_state:
                with open(upload_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                st.session_state.saved_upload = True
                st.session_state.upload_path = upload_path

            st.audio(st.session_state.upload_path)
    if mode == "Recording":
        st.title("Audio Recorder")
        audio = audiorecorder("Click to record", "Click to stop recording")

        if len(audio) > 0:
            st.audio(audio.export().read())

            if not st.session_state.saved_record:
                record_path = os.path.join(
                    UPLOAD_DIR,
                    f"{st.session_state.upload_id}_record.wav"
                )
                audio.export(record_path, format="wav")

                st.session_state.record_path = record_path
                st.session_state.saved_record = True

                st.write(f"Frame rate: {audio.frame_rate}, Frame width: {audio.frame_width}, Duration: {audio.duration_seconds} seconds")
st.write("---")
with st.container(border=True):
        st.subheader("2️⃣ Preprocessing Audio (Resample + Denoise)")
        st.info("Resampled input audio file to 16khz and de-noise with FFmpeg and Recurrent Neural Network")
        if st.button("Preprocessing audio files", use_container_width=True):
            if mode == "Upload file":
                preprocessing_wav_file(st.session_state.upload_path)
                audio = AudioSegment.from_file(st.session_state.upload_path)
                st.audio(st.session_state.upload_path)
                st.write(f"Frame rate: {audio.frame_rate}, Frame width: {audio.frame_width}, Duration: {audio.duration_seconds} seconds")
            elif mode == "Recording":
                preprocessing_wav_file(st.session_state.record_path)
                audio = AudioSegment.from_file(st.session_state.record_path)
                st.audio(st.session_state.record_path)
                st.write(f"Frame rate: {audio.frame_rate}, Frame width: {audio.frame_width}, Duration: {audio.duration_seconds} seconds")
            else:
                st.error("No audio file found!")
                st.stop()
with st.container(border=True):
        st.subheader("3️⃣ Speech Separation (MossFormer2)")
        st.info("Separate up to 2 overlapping speech sources from the input audio")
        if st.button("Speech Seperation", use_container_width=True):
            if mode == "Upload file" and st.session_state.upload_path:
                input_path = st.session_state.upload_path
                upload_output_folder_sp = os.path.join(UPLOAD_DIR, f"{st.session_state.upload_id}_upload")
                os.mkdir(upload_output_folder_sp)
                status = run_mossformer_inference(input_path, upload_output_folder_sp)
                st.text(status)
            elif mode == "Recording" and st.session_state.record_path:
                input_path = st.session_state.record_path
                record_output_folder_sp = os.path.join(UPLOAD_DIR, f"{st.session_state.upload_id}_record")
                os.mkdir(record_output_folder_sp)
                status = run_mossformer_inference(input_path, record_output_folder_sp)
                st.text(status)
with st.container(border=True):
        st.subheader("4️⃣ Speech-to-Text (Whisper Large)")
        st.info("Transcribe processed audio into text.")
        if st.button("Speech to Text", use_container_width=True):
            if mode == "Upload file" and st.session_state.upload_path:
                
                upload_output_folder_sp = os.path.join(UPLOAD_DIR, f"{st.session_state.upload_id}_upload")
                input_path_1 = os.path.join(upload_output_folder_sp, f"{st.session_state.upload_id}_upload_s1.wav")
                input_path_2 = os.path.join(upload_output_folder_sp, f"{st.session_state.upload_id}_upload_s2.wav")
                if os.path.exists(input_path_1) and os.path.exists(input_path_2):
                    text_1 = run_whisper_inference(input_path_1)
                    text_2 = run_whisper_inference(input_path_2)
                    st.text("Splited Text 1")
                    st.code(text_1)
                    st.text("Splited Text 2")
                    st.code(text_2)
                elif os.path.exists(st.session_state.upload_path):
                    text_1 = run_whisper_inference(upload_path)
                    st.text("Splited Text 1")
                    st.code(text_1)
            elif mode == "Recording" and st.session_state.record_path:
                record_output_folder_sp = os.path.join(UPLOAD_DIR, f"{st.session_state.upload_id}_record")
                input_path_1 = os.path.join(record_output_folder_sp, f"{st.session_state.upload_id}_record_s1.wav")
                input_path_2 = os.path.join(record_output_folder_sp, f"{st.session_state.upload_id}_record_s2.wav")
                if os.path.exists(input_path_1) and os.path.exists(input_path_2):
                    text_1 = run_whisper_inference(input_path_1)
                    text_2 = run_whisper_inference(input_path_2)
                    st.text("Splited Text 1")
                    st.code(text_1)
                    st.text("Splited Text 2")
                    st.code(text_2)
                elif os.path.exists(st.session_state.record_path):
                    text_1 = run_whisper_inference(record_path)
                    st.text("Splited Text 1")
                    st.code(text_1)