from transformers import T5ForConditionalGeneration, T5Tokenizer

# Load the pre-trained T5 model and tokenizer
model_name = "t5-small"  # You can also use 't5-base' or 't5-large' for more powerful models
model = T5ForConditionalGeneration.from_pretrained(model_name)
tokenizer = T5Tokenizer.from_pretrained(model_name)

# Function to perform text-to-text generation
def generate_text(input_text, task="summarize"):
    # Define the task-specific prefix (e.g., summarization, translation)
    input_text = f"{task}: " + input_text

    # Tokenize the input text
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids
    
    # Generate the output text
    output_ids = model.generate(input_ids, max_length=150, num_beams=5, early_stopping=True)
    
    # Decode the output into a human-readable string
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    return output_text

# Example input
input_text = "The Eiffel Tower is located in Paris, France, and is one of the most famous landmarks in the world. It was built in 1889 and stands 330 meters tall. Millions of people visit every year."

# Generate a summary (text-to-text generation)
output_text = generate_text(input_text, task="summarize")

# Print the result
print(f"Input Text: {input_text}")
print(f"Generated Output: {output_text}")

