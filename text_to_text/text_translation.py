from transformers import MarianMTModel, MarianTokenizer

# Function to translate text from one language to another
def translate_text(input_text, source_lang="en", target_lang="fr"):
    # Set up the translation model and tokenizer
    model_name = f'Helsinki-NLP/opus-mt-{source_lang}-{target_lang}'
    
    # Load the model and tokenizer
    model = MarianMTModel.from_pretrained(model_name)
    tokenizer = MarianTokenizer.from_pretrained(model_name)

    # Tokenize the input text
    inputs = tokenizer(input_text, return_tensors="pt", padding=True)

    # Generate translation
    translated = model.generate(**inputs)

    # Decode the generated tokens to text
    translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)
    
    return translated_text

# Example input (English)
input_text = "Hello, how are you?"

# Translate from English to French
translated_text = translate_text(input_text, source_lang="en", target_lang="fr")

# Print the original and translated text
print(f"Original: {input_text}")
print(f"Translated: {translated_text}")
