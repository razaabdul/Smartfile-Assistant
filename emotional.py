import gradio as gr
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import speech_recognition as sr

# Create an instance of the Recognizer class
recognizer = sr.Recognizer()

# Function to recognize speech from an audio file
def recognize_audio_file(file_path):
    with sr.AudioFile(file_path) as source:
        print("Loading audio file...")
        audio_data = recognizer.record(source)  # Read the entire audio file
        print("Done loading audio file.")
    
    try:
        print("Recognizing speech...")
        text = recognizer.recognize_google(audio_data, language='en-US')
        print('Your message: {}'.format(text))
        return text
    except Exception as ex:
        print("Error recognizing speech: {}".format(ex))
        return None

# Main function
def main():
    # Specify the path to your audio file
    audio_file_path = input("Please enter the path to your audio file: ")
    
    # Recognize speech from the audio file
    text = recognize_audio_file(audio_file_path)
    
    if text:
        # Sentiment analysis
        analyser = SentimentIntensityAnalyzer()
        v = analyser.polarity_scores(text)
        print("Sentiment analysis results:", v)

if __name__ == "__main__":
    main()
