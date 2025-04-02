import os
import gradio as gr

from brain_of_dcotor import encode_image, analyze_image_with_query
from voice_of_patient import record_audio, transcribe_with_groq
from voice_of_doctor import text_to_speech_with_gtts, text_to_speech_with_elevenlabs


def process_inputs():
    pass


iface=gr.Interface(
    fn=process_inputs(),
    inputs=[
        gr.aud
    ]
)