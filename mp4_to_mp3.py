from moviepy import VideoFileClip

# Load the video file (using raw string or double backslashes)
video = VideoFileClip(r"C:\Users\hp\Desktop\RAG\cam.mp4")

# Extract audio and write it to an MP3 file (using raw string or double backslashes)
video.audio.write_audiofile(r"C:\Users\hp\Desktop\RAG\output_material.mp3")


