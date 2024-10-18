from gtts import gTTS
import os

# Text you want to convert to speech
text = "Hello, welcome to the world of Text to Speech!"

# Initialize TTS
tts = gTTS(text=text, lang='en')

# Save the audio file
tts.save("output.mp3")

# Play the audio file using a Linux media player like mpg321, ffplay, or aplay
os.system("mpg321 output.mp3")  # mpg321 needs to be installed
# os.system("ffplay -autoexit output.mp3")  # Use this if you have ffmpeg installed
# os.system("aplay output.mp3")  # Works for .wav files, convert mp3 to wav if needed
