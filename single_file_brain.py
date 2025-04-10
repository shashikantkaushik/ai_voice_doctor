# if you dont use pipenv uncomment the following:
# from dotenv import load_dotenv
# load_dotenv()

# Step 1: Setup environment and imports
import os
import gradio as gr
import base64
import logging
import speech_recognition as sr
from pydub import AudioSegment
from io import BytesIO
import subprocess
import platform
import time
from groq import Groq
from gtts import gTTS
import elevenlabs
from elevenlabs.client import ElevenLabs
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
ELEVENLABS_API_KEY = os.environ.get("ELEVENLABS_API_KEY")

# Available Groq models
VISION_MODEL = "llama-3.2-90b-vision-preview"  # Vision model for image analysis
TEXT_MODEL = "llama3-8b-8192"  # Text-only model for conversation


# Image processing functions
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def analyze_image_with_query(query, model, encoded_image):
    client = Groq(api_key=GROQ_API_KEY)
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": query
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{encoded_image}",
                    },
                },
            ],
        }
    ]
    chat_completion = client.chat.completions.create(
        messages=messages,
        model=model
    )
    return chat_completion.choices[0].message.content


# Audio recording functions
def record_audio(file_path, timeout=20, phrase_time_limit=None):
    recognizer = sr.Recognizer()
    try:
        with sr.Microphone() as source:
            logging.info("Adjusting for ambient noise...")
            recognizer.adjust_for_ambient_noise(source, duration=1)
            logging.info("Start speaking now...")
            audio_data = recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
            logging.info("Recording complete.")
            wav_data = audio_data.get_wav_data()
            audio_segment = AudioSegment.from_wav(BytesIO(wav_data))
            audio_segment.export(file_path, format="mp3", bitrate="128k")
            logging.info(f"Audio saved to {file_path}")
    except Exception as e:
        logging.error(f"An error occurred: {e}")


# Speech to text function
def transcribe_with_groq(audio_filepath, stt_model="whisper-large-v3", GROQ_API_KEY=None):
    client = Groq(api_key=GROQ_API_KEY)
    with open(audio_filepath, "rb") as audio_file:
        transcription = client.audio.transcriptions.create(
            model=stt_model,
            file=audio_file,
            language="en"
        )
    return transcription.text


# Text to speech functions
def text_to_speech_with_gtts(input_text, output_filepath):
    audioobj = gTTS(
        text=input_text,
        lang="en",
        slow=False
    )
    audioobj.save(output_filepath)
    play_audio(output_filepath)
    return output_filepath


def text_to_speech_with_elevenlabs(input_text, output_filepath):
    client = ElevenLabs(api_key=ELEVENLABS_API_KEY)
    audio = client.generate(
        text=input_text,
        voice="Aria",
        output_format="mp3_22050_32",
        model="eleven_turbo_v2"
    )
    elevenlabs.save(audio, output_filepath)
    play_audio(output_filepath)
    return output_filepath


def play_audio(filepath):
    os_name = platform.system()
    try:
        if os_name == "Darwin":  # macOS
            subprocess.run(['afplay', filepath])
        elif os_name == "Windows":  # Windows
            subprocess.run(['powershell', '-c', f'(New-Object Media.SoundPlayer "{filepath}").PlaySync();'])
        elif os_name == "Linux":  # Linux
            subprocess.run(['aplay', filepath])  # Alternative: use 'mpg123' or 'ffplay'
        else:
            raise OSError("Unsupported operating system")
    except Exception as e:
        print(f"An error occurred while trying to play the audio: {e}")


# Conversation state management
class ConversationState:
    def __init__(self):
        self.conversation_history = []
        self.current_image = None
        self.conversation_stage = "initial"  # initial, gathering_info, diagnosis
        self.follow_up_questions = [
            "Can you describe your symptoms in more detail?",
            "How long have you been experiencing these symptoms?",
            "Have you tried any remedies so far?",
            "Do you have any allergies or medical conditions I should know about?",
            "Is there any family history related to this condition?",
            "Are you currently taking any medications?",
            "On a scale of 1-10, how severe is your pain?",
            "Have you noticed any triggers that make your symptoms worse?",
            "Have you had any fever or chills with these symptoms?",
            "Have you experienced any recent weight changes?"
        ]
        self.question_index = 0
        self.is_doctor_speaking = False
        self.patient_info = {
            "symptoms": [],
            "duration": "",
            "severity": "",
            "allergies": [],
            "medications": [],
            "medical_history": []
        }

    def add_message(self, role, content):
        if content:  # Only add if there's actual content
            self.conversation_history.append({"role": role, "content": content})

    def get_conversation_for_ai(self):
        system_prompt = """You are Dr. Smith, an experienced AI medical assistant. You are conducting a patient consultation with professional yet empathetic bedside manner.

        Your consultation should follow this structure:
        1. Greet the patient warmly and establish rapport
        2. Gather comprehensive medical information:
           - Current symptoms (onset, duration, severity)
           - Medical history (allergies, conditions, medications)
           - Lifestyle factors (diet, exercise, sleep)
           - Family medical history
        3. Analyze any provided visual symptoms
        4. Provide preliminary assessment with possible diagnoses
        5. Suggest appropriate remedies (OTC medications, lifestyle changes)
        6. Recommend when to seek in-person care

        Key behaviors:
        - Ask one question at a time
        - Use layman's terms but maintain professionalism
        - Show empathy and reassurance
        - For medications, always include dosage and precautions
        - When suggesting OTC meds, recommend specific names (e.g., "take 200-400mg ibuprofen every 4-6 hours")
        - Always ask about allergies before recommending medications
        - For serious symptoms, clearly advise immediate medical attention

        Current patient information:
        Symptoms: {symptoms}
        Duration: {duration}
        Severity: {severity}
        Allergies: {allergies}
        Medications: {medications}
        Medical History: {medical_history}
        """.format(**self.patient_info)

        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(self.conversation_history)
        return messages

    def get_next_question(self):
        if self.question_index < len(self.follow_up_questions):
            question = self.follow_up_questions[self.question_index]
            self.question_index += 1
            return question
        else:
            return "Based on the information you've provided, I can now give you my assessment."

    def update_patient_info(self, text_input):
        """Extract patient information from conversation"""
        # This is a simplified version - in practice you'd use more sophisticated NLP
        text = text_input.lower()

        # Extract duration phrases
        duration_phrases = ["for about", "for around", "since", "for the past", "for last"]
        if any(phrase in text for phrase in duration_phrases):
            self.patient_info["duration"] = text_input

        # Extract severity
        if "scale of 1 to 10" in text or "1-10" in text:
            self.patient_info["severity"] = text_input

        # Extract allergies
        if "allerg" in text or "react" in text:
            self.patient_info["allergies"].append(text_input)

        # Extract medications
        med_keywords = ["take", "using", "prescribed", "medication", "pill"]
        if any(keyword in text for keyword in med_keywords):
            self.patient_info["medications"].append(text_input)

    def reset(self):
        self.__init__()

    def set_doctor_speaking(self, is_speaking):
        self.is_doctor_speaking = is_speaking
        return self.is_doctor_speaking


# Initialize conversation state
conversation_state = ConversationState()


def handle_error(error_message):
    """Handle errors gracefully with user-friendly messages"""
    logging.error(f"Error: {error_message}")
    return f"I apologize, but I encountered an error: {error_message}. Please try again."


# Generate AI response based on conversation history
def generate_ai_response(text_input=None, image_input=None):
    client = Groq(api_key=GROQ_API_KEY)

    try:
        if text_input:
            conversation_state.update_patient_info(text_input)

        if image_input and conversation_state.conversation_stage == "initial":
            # First interaction with image
            conversation_state.current_image = image_input
            encoded_image = encode_image(image_input)

            image_prompt = """Analyze this medical image carefully. As Dr. Smith, please:
            1. Describe any visible abnormalities in layman's terms
            2. Ask relevant follow-up questions about:
               - Symptom duration
               - Pain level
               - Any treatments tried
               - Medical history
            3. Maintain professional but empathetic tone
            4. Keep response to 2-3 sentences ending with a question

            The patient said: {text_input}""".format(text_input=text_input if text_input else "")

            response = analyze_image_with_query(
                query=image_prompt,
                encoded_image=encoded_image,
                model=VISION_MODEL
            )

            conversation_state.add_message("user", text_input if text_input else "")
            conversation_state.add_message("assistant", response)
            conversation_state.conversation_stage = "gathering_info"

        elif text_input:
            # Text-based conversation
            conversation_state.add_message("user", text_input)

            if conversation_state.conversation_stage == "initial":
                # No image, first interaction
                conversation_state.conversation_stage = "gathering_info"
                next_question = """Hello, I'm Dr. Smith. Thank you for consulting with me today. 
                To help you best, I'll need to ask some questions about your symptoms. 
                What health concerns bring you in today?"""
                conversation_state.add_message("assistant", next_question)
                response = next_question
            else:
                # Ongoing conversation
                messages = conversation_state.get_conversation_for_ai()

                # Generate more detailed responses when in diagnosis stage
                if conversation_state.conversation_stage == "diagnosis":
                    messages.append({
                        "role": "system",
                        "content": "Now provide a detailed assessment including: 1) Possible conditions, 2) Recommended OTC medications with dosages, 3) Home care advice, 4) When to seek emergency care. Be specific about medication names and dosages."
                    })

                chat_completion = client.chat.completions.create(
                    messages=messages,
                    model=TEXT_MODEL,
                    max_tokens=1024  # Allow longer responses for diagnosis
                )
                response = chat_completion.choices[0].message.content
                conversation_state.add_message("assistant", response)

                # Check if we should transition to diagnosis
                if conversation_state.question_index >= len(conversation_state.follow_up_questions):
                    conversation_state.conversation_stage = "diagnosis"
        else:
            # No input provided yet
            response = """Hello, I'm Dr. Smith. I'll be assisting you today. 
            Please describe your symptoms or upload an image of any visible concerns."""
            conversation_state.add_message("assistant", response)

        return response

    except Exception as e:
        return handle_error(str(e))


# Process both audio and image inputs
def process_inputs(audio_filepath=None, image_filepath=None):
    text_input = None

    # Update UI to show processing state
    yield text_input, "Processing your input...", None, "", True

    try:
        # Process audio to text if provided
        if audio_filepath:
            text_input = transcribe_with_groq(
                GROQ_API_KEY=GROQ_API_KEY,
                audio_filepath=audio_filepath,
                stt_model="whisper-large-v3"
            )

        # Generate AI response
        doctor_response = generate_ai_response(text_input, image_filepath)

        # Update UI to show doctor is speaking
        yield text_input, doctor_response, None, format_conversation_history(), True

        # Convert response to speech
        output_filepath = f"response_{len(conversation_state.conversation_history)}.mp3"
        voice_output = text_to_speech_with_elevenlabs(input_text=doctor_response, output_filepath=output_filepath)

        # Final update with audio
        yield text_input, doctor_response, voice_output, format_conversation_history(), False

    except Exception as e:
        error_msg = handle_error(str(e))
        yield text_input if text_input else "Error processing audio", error_msg, None, "", False


def format_conversation_history():
    """Format conversation history for display with styling"""
    conversation_html = "<div style='display: flex; flex-direction: column; gap: 12px;'>"

    for i in range(0, len(conversation_state.conversation_history), 2):
        if i < len(conversation_state.conversation_history):
            user_message = conversation_state.conversation_history[i]['content']
            conversation_html += f"""
            <div style='align-self: flex-end; 
                        max-width: 80%; 
                        background: linear-gradient(135deg, #2b5876 0%, #4e4376 100%); 
                        padding: 12px 16px; 
                        border-radius: 18px 18px 0 18px; 
                        color: #e0e0e0;
                        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);'>
                <b style='color: #7fdbff;'>You:</b> {user_message}
            </div>
            """

        if i + 1 < len(conversation_state.conversation_history):
            ai_message = conversation_state.conversation_history[i + 1]['content']
            conversation_html += f"""
            <div style='align-self: flex-start; 
                        max-width: 80%; 
                        background: linear-gradient(135deg, #434343 0%, #000000 100%); 
                        padding: 12px 16px; 
                        border-radius: 18px 18px 18px 0; 
                        color: #f0f0f0;
                        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);'>
                <b style='color: #7fdbff;'>Dr. Smith:</b> {ai_message}
            </div>
            """

    conversation_html += "</div>"
    return conversation_html


def reset_conversation():
    conversation_state.reset()
    return None, "Consultation has been reset. How can I help you today?", None, "", False


# Custom CSS for the dark mode interface
dark_mode_css = """
:root {
    --primary: #7fdbff;
    --secondary: #2ecc40;
    --accent: #ff851b;
    --dark-bg: #121212;
    --darker-bg: #0a0a0a;
    --card-bg: #1e1e1e;
    --text-primary: #f0f0f0;
    --text-secondary: #b0b0b0;
    --border-radius: 12px;
    --spacing: 16px;
}

.gradio-container {
    font-family: 'Inter', system-ui, -apple-system, sans-serif;
    background: var(--dark-bg) !important;
    color: var(--text-primary) !important;
    min-height: 100vh;
    padding: 0 !important;
    margin: 0 !important;
}

.main-container {
    max-width: none !important;
    margin: 0 !important;
    padding: 0 !important;
    height: 100vh;
    display: grid;
    grid-template-columns: 1fr 2fr;
    gap: var(--spacing);
}

.header {
    background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
    color: white;
    padding: 1.5rem;
    text-align: center;
    border-radius: 0;
    margin-bottom: 0;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    grid-column: 1 / -1;
}

.header h1 {
    margin: 0;
    font-size: 1.8rem;
    font-weight: 600;
    letter-spacing: 0.5px;
}

.header p {
    margin: 0.5rem 0 0;
    opacity: 0.9;
    font-size: 0.95rem;
}

.control-panel {
    background: var(--card-bg);
    padding: var(--spacing);
    border-radius: var(--border-radius);
    margin: var(--spacing);
    height: calc(100vh - 180px);
    display: flex;
    flex-direction: column;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
    border: 1px solid rgba(255, 255, 255, 0.05);
}

.conversation-area {
    background: var(--card-bg);
    padding: var(--spacing);
    border-radius: var(--border-radius);
    margin: var(--spacing);
    height: calc(100vh - 180px);
    display: flex;
    flex-direction: column;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
    border: 1px solid rgba(255, 255, 255, 0.05);
}

.conversation-display {
    flex-grow: 1;
    overflow-y: auto;
    padding: var(--spacing);
    margin-bottom: var(--spacing);
    background: var(--darker-bg);
    border-radius: var(--border-radius);
    border: 1px solid rgba(255, 255, 255, 0.05);
}

.image-upload {
    border: 2px dashed rgba(127, 219, 255, 0.3) !important;
    border-radius: var(--border-radius) !important;
    background: var(--darker-bg) !important;
    margin-bottom: var(--spacing) !important;
    min-height: 100px;
}

.image-upload:hover {
    border-color: var(--primary) !important;
}

.audio-input {
    width: 100% !important;
    margin-bottom: var(--spacing) !important;
}

button {
    background: linear-gradient(135deg, #0f2027 0%, #203a43 100%) !important;
    color: white !important;
    border: none !important;
    padding: 10px 20px !important;
    border-radius: 8px !important;
    font-weight: 500 !important;
    transition: all 0.2s ease !important;
    text-transform: uppercase !important;
    letter-spacing: 0.5px !important;
    font-size: 0.85rem !important;
    margin-right: 10px !important;
}

button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3) !important;
}

button:active {
    transform: translateY(0) !important;
}

button.primary {
    background: linear-gradient(135deg, #0074e4 0%, #00a1ff 100%) !important;
}

button.reset-btn {
    background: linear-gradient(135deg, #ff416c 0%, #ff4b2b 100%) !important;
}

.user-message, .doctor-response {
    background: var(--darker-bg) !important;
    color: var(--text-primary) !important;
    border: 1px solid rgba(255, 255, 255, 0.05) !important;
    border-radius: var(--border-radius) !important;
    padding: var(--spacing) !important;
    margin-bottom: var(--spacing) !important;
    width: calc(100% - 40px) !important;
}

.doctor-audio {
    width: calc(100% - 40px) !important;
    margin-top: var(--spacing) !important;
}

/* Scrollbar styling */
::-webkit-scrollbar {
    width: 8px;
}

::-webkit-scrollbar-track {
    background: rgba(255, 255, 255, 0.05);
    border-radius: 4px;
}

::-webkit-scrollbar-thumb {
    background: rgba(127, 219, 255, 0.3);
    border-radius: 4px;
}

::-webkit-scrollbar-thumb:hover {
    background: var(--primary);
}

/* Animation for doctor speaking */
@keyframes pulse-glow {
    0% {
        box-shadow: 0 0 0 0 rgba(127, 219, 255, 0.7);
    }
    70% {
        box-shadow: 0 0 0 10px rgba(127, 219, 255, 0);
    }
    100% {
        box-shadow: 0 0 0 0 rgba(127, 219, 255, 0);
    }
}

.doctor-speaking .doctor-response {
    animation: pulse-glow 1.5s infinite;
    border-color: var(--primary) !important;
}

/* Microphone active state */
.microphone-active {
    background: #ff416c !important;
    box-shadow: 0 0 10px #ff416c !important;
}

/* Input labels */
label {
    color: var(--text-primary) !important;
    margin-bottom: 8px !important;
    font-weight: 500 !important;
    font-size: 0.9rem !important;
}

/* Gradio's dark mode overrides */
.dark input, .dark textarea, .dark select {
    background: var(--darker-bg) !important;
    color: var(--text-primary) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
}

.dark input:focus, .dark textarea:focus {
    border-color: var(--primary) !important;
    box-shadow: 0 0 0 2px rgba(127, 219, 255, 0.2) !important;
}

/* Responsive adjustments */
@media (max-width: 768px) {
    .main-container {
        grid-template-columns: 1fr;
    }

    .control-panel, .conversation-area {
        height: auto;
        min-height: 300px;
    }
}
"""

# Create the Gradio interface with dark mode design
with gr.Blocks(title="AI Doctor with Vision and Voice", css=dark_mode_css) as iface:
    # Header section
    with gr.Row(elem_classes="header"):
        gr.HTML("""
        <div>
            <h1>AI Medical Consultation</h1>
            <p>Virtual consultation with Dr. Smith - Describe symptoms or upload images</p>
        </div>
        """)

    # Main content area
    with gr.Row(elem_classes="main-container"):
        # Left panel - controls
        with gr.Column(elem_classes="control-panel"):
            image_input = gr.Image(type="filepath", label="Upload Medical Image", elem_classes="image-upload")
            audio_input = gr.Audio(
                sources=["microphone"],
                type="filepath",
                label="Record Your Symptoms",
                elem_classes="audio-input"
            )

            with gr.Row():
                submit_btn = gr.Button("Submit", variant="primary")
                reset_btn = gr.Button("Reset Consultation", variant="stop", elem_classes="reset-btn")

        # Right panel - conversation
        with gr.Column(elem_classes="conversation-area"):
            # Hidden checkbox for doctor speaking state
            doctor_speaking = gr.Checkbox(label="Doctor is speaking", visible=False)

            # Conversation display
            conversation_display = gr.HTML(
                label="Consultation History",
                elem_classes="conversation-display",
                value="<div style='text-align: center; padding: 20px; color: var(--text-secondary);'>Your consultation will appear here</div>"
            )

            # Response area (now properly contained within the layout)
            with gr.Column(visible=False) as response_container:
                speech_to_text_output = gr.Textbox(
                    label="Your Message",
                    visible=False
                )
                doctor_response_output = gr.Textbox(
                    label="Doctor's Response",
                    elem_classes="doctor-response"
                )
                audio_output = gr.Audio(
                    label="Doctor's Voice Response",
                    elem_classes="doctor-audio",
                    visible=True
                )

    # Event handlers
    submit_btn.click(
        fn=process_inputs,
        inputs=[audio_input, image_input],
        outputs=[speech_to_text_output, doctor_response_output, audio_output, conversation_display, doctor_speaking]
    ).then(
        lambda: gr.update(visible=True),
        outputs=[response_container]
    )

    reset_btn.click(
        fn=reset_conversation,
        inputs=[],
        outputs=[image_input, doctor_response_output, audio_output, conversation_display, doctor_speaking]
    ).then(
        lambda: gr.update(visible=False),
        outputs=[response_container]
    )


    # Doctor speaking state handler
    def update_speaking_state(speaking):
        return gr.update(interactive=not speaking)


    doctor_speaking.change(
        fn=update_speaking_state,
        inputs=[doctor_speaking],
        outputs=[submit_btn]
    )

# Launch the interface
if __name__ == "__main__":
    iface.launch(
        debug=True,
        server_name="0.0.0.0",
        server_port=7862,
        share=False,
        favicon_path=None,
        inbrowser=True
    )