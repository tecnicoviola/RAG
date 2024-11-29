from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("suno/bark-small")
model = AutoModelForSeq2SeqLM.from_pretrained("suno/bark-small")

input_text = "Your input text"
inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)

# Now, you can add attention_mask and pad_token_id explicitly:
inputs["attention_mask"] = inputs.get("attention_mask", torch.ones_like(inputs["input_ids"]))
inputs["pad_token_id"] = tokenizer.pad_token_id  # Default pad_token_id

outputs = model.generate(**inputs)
