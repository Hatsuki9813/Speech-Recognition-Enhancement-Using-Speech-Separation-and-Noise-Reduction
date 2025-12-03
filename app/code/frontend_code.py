import sys, os
import streamlit as st
import pandas as pd
from io import StringIO
import tempfile
import uuid
from audiorecorder import audiorecorder
from backend_code import preprocessing_wav_file, run_mossformer_inference, run_whisper_inference, normalize_text
UPLOAD_DIR = "../audio"
if "upload_id" not in st.session_state:
    st.session_state.upload_id = str(uuid.uuid4())
if "record_path" not in st.session_state:
    st.session_state.record_path = None
if "saved_record" not in st.session_state:
    st.session_state.saved_record = False
st.set_page_config(layout="wide")

st.markdown("<h1 style='text-align: center; '>MOSSFORMER2 - WHISPER SYSTEM</h1>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center; '>This is a demo of a combination of FFmpeg noise reduction using rnnn, MossFormer2 voice sepearation and Whisper speech to text voice recogniton</h2>", unsafe_allow_html=True)
with st.container(border=True):
    mode = st.radio(
        "Select input way ",
        ("Upload file", "Recording")
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
with st.container(border=True):
    
        if st.button("Preprocessing audio files", use_container_width=True):
            if mode == "Upload file":
                input_path = st.session_state.upload_path
            elif mode == "Recording":
                input_path = st.session_state.record_path
            else:
                st.error("No audio file found!")
                st.stop()
            status = preprocessing_wav_file(input_path)
            if status:
                st.success("Finished processing")
            else:
                st.warning("Processing failed!")


with st.container(border=True):

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

        if st.button("Speech to Text", use_container_width=True):
            if mode == "Upload file" and st.session_state.upload_path:
                upload_output_folder_sp = os.path.join(UPLOAD_DIR, f"{st.session_state.upload_id}_upload")
                input_path_1 = os.path.join(upload_output_folder_sp, f"{st.session_state.upload_id}_s1.wav")
                input_path_2 = os.path.join(upload_output_folder_sp, f"{st.session_state.upload_id}_s2.wav")
                text_1 = run_whisper_inference(input_path_1)
                text_2 = run_whisper_inference(input_path_2)
                st.text("Splited Text 1")
                st.code(text_1)
                st.text("Splited Text 2")
                st.code(text_2)
            elif mode == "Recording" and st.session_state.record_path:
                record_output_folder_sp = os.path.join(UPLOAD_DIR, f"{st.session_state.upload_id}_record")
                input_path_1 = os.path.join(record_output_folder_sp, f"{st.session_state.upload_id}_record_s1.wav")
                input_path_2 = os.path.join(record_output_folder_sp, f"{st.session_state.upload_id}_record_s2.wav")
                text_1 = run_whisper_inference(input_path_1)
                text_2 = run_whisper_inference(input_path_2)
                st.text("Splited Text 1")
                st.code(text_1)
                st.text("Splited Text 2")
                st.code(text_2)