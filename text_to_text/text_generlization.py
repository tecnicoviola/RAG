from transformers import pipeline

# Load the paraphrasing model
paraphraser = pipeline("text2text-generation", model="t5-base")

# Input text
input_text = "The quick brown fox jumps over the lazy dog."

# Generate paraphrased text
paraphrased_text = paraphraser(input_text)
print("Paraphrased Text:", paraphrased_text[0]['generated_text'])
