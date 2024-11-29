from transformers import pipeline

# Load the summarization pipeline from Hugging Face
summarizer = pipeline("summarization")

# Input text
input_text = """
Artificial intelligence (AI) is intelligence demonstrated by machines, in contrast to the natural intelligence displayed by humans and animals. 
Leading AI textbooks define the field as the study of "intelligent agents": any device that perceives its environment and takes actions that maximize its chance of successfully achieving its goals.
"""

# Summarize the input text
summary = summarizer(input_text)
print("Summary:", summary[0]['summary_text'])
   