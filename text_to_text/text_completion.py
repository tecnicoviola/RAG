from transformers import GPT2LMHeadModel, GPT2Tokenizer

model_name = "gpt2"  # Replace with the model you're using
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Example input text
input_text = "Once upon a time in a faraway land, there was a"

# Tokenize the input text
inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)

# Manually set pad_token_id (if it's not set)
model.config.pad_token_id = model.config.eos_token_id  # If you want to use eos_token_id as pad_token_id

# Optionally, you can also set the attention_mask manually
inputs['attention_mask'] = inputs.get('attention_mask', torch.ones(inputs['input_ids'].shape))

# Generate text
output = model.generate(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'], max_length=50)

# Decode the output
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

print(generated_text)
