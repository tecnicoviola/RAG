from transformers import T5ForConditionalGeneration, T5Tokenizer

# Load the pre-trained T5 model and tokenizer
model_name = "t5-small"  # You can also use 't5-base' or 't5-large' for larger models
model = T5ForConditionalGeneration.from_pretrained(model_name)
tokenizer = T5Tokenizer.from_pretrained(model_name)

# Function to paraphrase input text
def paraphrase_text(input_text):
    # Add the task-specific prefix for paraphrasing
    input_text = "paraphrase: " + input_text

    # Tokenize the input text
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids
    
    # Generate the paraphrased text
    output_ids = model.generate(input_ids, max_length=150, num_beams=5, early_stopping=True)
    
    # Decode the generated text
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    return output_text

# Example input text
input_text = "The quick brown fox jumps over the lazy dog."

# Paraphrase the input text
paraphrased_text = paraphrase_text(input_text)

# Print the original and paraphrased text
print(f"Original: {input_text}")
print(f"Paraphrased: {paraphrased_text}")
