from gtts import gTTS
import os

# Function to convert text to speech
def text_to_speech(text, language='en', output_file="output.mp3"):
    # Create a gTTS object
    tts = gTTS(text=text, lang=language, slow=False)
    
    # Save the speech as an MP3 file
    tts.save(output_file)
    
    # Play the speech
    os.system(f"start {output_file}")  # On Windows
    # os.system(f"mpg321 {output_file}")  # On Linux
    # os.system(f"afplay {output_file}")  # On macOS

# Example usage
text = "Hello, this is a text to speech conversion example."
text_to_speech(text)
