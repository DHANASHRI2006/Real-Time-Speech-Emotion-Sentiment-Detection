**🎙️ Real-Time Speech Emotion & Sentiment Detection**

This project is a real-time English speech transcription system with integrated sentiment and emotion detection, powered by NLP transformer models, Google Speech Recognition, and a Streamlit interface.

**📖 Project Description**

The application allows users to:

🎤 Transcribe live audio from the microphone or upload .wav audio files

✍️ Restore proper punctuation using a multilingual punctuation model

😊 Analyze emotions (like Joy, Sadness, Anger) based on context

👍 Perform sentiment classification (Positive, Negative)

📊 Ensure consistent interpretation by aligning sentiment with emotion

All results are displayed live in an interactive Streamlit UI, providing a real-time emotional overview of spoken content.


**🔍 Key Features**
✅ Live microphone input and audio file upload support

✅ Real-time speech-to-text transcription (via Google Speech API)

✅ Automatic punctuation restoration with multilingual model

✅ Accurate sentiment analysis using Hugging Face pipelines

✅ Deep emotion detection using DistilRoBERTa emotion classifier

✅ Streamlit interface with live updates and visual display

**🧠 Models and Tools Used**

Component	Library/Model
Speech Recognition	speech_recognition (Google Web Speech API)
Sentiment Analysis	transformers - pipeline("sentiment-analysis")
Emotion Detection	j-hartmann/emotion-english-distilroberta-base (via Hugging Face)
Punctuation Restore	deepmultilingualpunctuation - oliverguhr/fullstop-punctuation-multilang
UI Framework	streamlit

**▶️ How to Run**

1. 🚀 Start the Application

streamlit run speech_module.py


**✅ Advantages**

🔄 Real-time audio-to-text processing

🧠 Deeper emotional context than basic transcription

🔍 Consistent emotion-sentiment alignment logic

🖱️ No code knowledge required – fully GUI-based

🌍 Can be adapted to other languages or domains

