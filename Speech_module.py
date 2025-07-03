import streamlit as st
import speech_recognition as sr
from transformers import pipeline
import time
from deepmultilingualpunctuation import PunctuationModel
# ✅ Load Sentiment and Emotion Pipelines
@st.cache_resource
def load_pipelines():
    # Sentiment model
    sentiment = pipeline("sentiment-analysis")
    # Emotion detection using DistilRoBERTa-base
    emotion = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=True)
    return sentiment, emotion   

sentiment_analyzer, emotion_analyzer = load_pipelines()

# ✅ Load Punctuation Model
@st.cache_resource
def load_punctuation_model():
    try:
        model = PunctuationModel(model="oliverguhr/fullstop-punctuation-multilang-large")  
        return model
    except Exception as e:
        st.error(f"Error loading punctuation model: {e}")
        return None

punctuation_model = load_punctuation_model()

# ✅ Sentiment Analysis
def analyze_sentiment(text):
    try:
        result = sentiment_analyzer(text)[0]
        return result['label'], result['score']
    except Exception as e:
        return "Error", f"Could not analyze sentiment: {e}"

# ✅ Consistent Emotion Detection with Sentiment Matching
def analyze_consistent_emotion(text, sentiment_label):
    try:
        emotions = emotion_analyzer(text)
        emotion_scores = {e['label']: e['score'] for e in emotions[0]}

        # ✅ Categorize emotions into Positive and Negative
        positive_emotions = {k: v for k, v in emotion_scores.items() if k in ["joy", "surprise"]}
        negative_emotions = {k: v for k, v in emotion_scores.items() if k in ["anger", "disgust", "fear", "sadness"]}

        # ✅ Filter emotions based on sentiment
        if sentiment_label == "POSITIVE":
            relevant_emotions = positive_emotions
        elif sentiment_label == "NEGATIVE":
            relevant_emotions = negative_emotions
        else:
            relevant_emotions = emotion_scores  # Neutral or mixed cases

        # ✅ Identify dominant emotion
        dominant_emotion = max(relevant_emotions, key=relevant_emotions.get, default="neutral")
        dominant_score = relevant_emotions.get(dominant_emotion, 0.0)

        return dominant_emotion, dominant_score

    except Exception as e:
        return "Error", 0.0

# ✅ Punctuation Restoration
def restore_punctuation(text):
    if punctuation_model:
        try:
            result = punctuation_model.restore_punctuation(text)
            return result.strip()
        except Exception as e:
            st.warning(f"⚠️ Punctuation restoration failed: {e}")
            return text.strip() + "."
    else:
        return text.strip() + "."

# ✅ Live Audio Transcription with Consistent Sentiment and Emotion Analysis
def transcribe_live_audio(live_transcription_placeholder, sentiment_placeholder, emotion_placeholder):
    recognizer = sr.Recognizer()
    mic = sr.Microphone()

    with mic as source:
        st.info("🎤 Listening for live audio...")
        recognizer.adjust_for_ambient_noise(source)

        st.session_state['live_transcript'] = ""  

        while st.session_state.get("live_listening", False):
            try:
                audio = recognizer.listen(source, timeout=1)
                text = recognizer.recognize_google(audio, language="en-IN")
                st.session_state['live_transcript'] += f" {text}"
                
                # Punctuation restoration
                punctuated_text = restore_punctuation(st.session_state['live_transcript'].strip())
                live_transcription_placeholder.markdown(f"**Live:**\n\n{punctuated_text}")
                
                # Sentiment Analysis
                if text:
                    sentiment_label, sentiment_score = analyze_sentiment(text)
                    sentiment_placeholder.info(f"💬 Sentiment: {sentiment_label} ({sentiment_score:.2f})")
                    
                    # Consistent Emotion Detection
                    emotion_label, emotion_score = analyze_consistent_emotion(text, sentiment_label)
                    emotion_placeholder.info(f"😃 Emotion: {emotion_label.capitalize()} ({emotion_score:.2f})")

            except sr.UnknownValueError:
                st.warning("⚠️ Could not understand audio")
            except sr.RequestError as e:
                st.error(f"⚠️ Could not request results. Check internet connection. Error: {e}")
            except sr.WaitTimeoutError:
                pass 
            except Exception as e:
                st.error(f"❌ An unexpected error occurred: {e}")
                break
            time.sleep(0.01)

# ✅ File-based Transcription with Consistent Sentiment and Emotion Analysis
def transcribe_file_audio(uploaded_file, file_transcription_placeholder, file_sentiment_placeholder, file_emotion_placeholder):
    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(uploaded_file) as source:
            audio_data = recognizer.record(source)

        # Transcribe audio
        text = recognizer.recognize_google(audio_data, language="en-IN")
        
        # Punctuate
        punctuated_text = restore_punctuation(text)
        file_transcription_placeholder.markdown(f"**Transcription:**\n\n{punctuated_text}")

        # Sentiment Analysis
        sentiment_label, sentiment_score = analyze_sentiment(text)
        file_sentiment_placeholder.info(f"💬 Sentiment: {sentiment_label} ({sentiment_score:.2f})")

        # Consistent Emotion Detection
        emotion_label, emotion_score = analyze_consistent_emotion(text, sentiment_label)
        file_emotion_placeholder.info(f"😃 Emotion: {emotion_label.capitalize()} ({emotion_score:.2f})")

    except sr.UnknownValueError:
        file_transcription_placeholder.warning("⚠️ Could not understand audio from file")
    except sr.RequestError as e:
        file_transcription_placeholder.error(f"⚠️ Check internet connection. Error: {e}")
    except Exception as e:
        file_transcription_placeholder.error(f"❌ Error processing the file: {e}")

# ✅ Main Function
def main():
    st.title("🎙️ Real-time English Audio Transcription with Consistent Sentiment & Emotion Detection")

    # ✅ Live Transcription
    st.header("🔴 Live Transcription")
    live_transcription_placeholder = st.empty()
    live_sentiment_placeholder = st.empty()
    live_emotion_placeholder = st.empty()

    start_live = st.button("▶️ Start Live Transcription", on_click=lambda: st.session_state.update({"live_listening": True, "live_transcript": ""}))
    stop_live = st.button("⏹️ Stop Live Transcription", on_click=lambda: st.session_state.update({"live_listening": False}))

    if "live_listening" not in st.session_state:
        st.session_state["live_listening"] = False
    if "live_transcript" not in st.session_state:
        st.session_state["live_transcript"] = ""

    if st.session_state.get("live_listening", False):
        transcribe_live_audio(live_transcription_placeholder, live_sentiment_placeholder, live_emotion_placeholder)

    # ✅ File Upload Transcription
    st.header("📁 Upload Audio File")
    uploaded_file = st.file_uploader("Upload an English audio file (WAV)", type=["wav"])
    
    file_transcription_placeholder = st.empty()
    file_sentiment_placeholder = st.empty()
    file_emotion_placeholder = st.empty()

    if uploaded_file is not None:
        transcribe_file_audio(uploaded_file, file_transcription_placeholder, file_sentiment_placeholder, file_emotion_placeholder)

def speech_app():
    main()  # assuming `main()` is your app function in the speech code
