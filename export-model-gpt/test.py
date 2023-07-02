from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load pre-trained model and tokenizer
model_name = "gpt2"  # You can also use "gpt2-medium" or "gpt2-large" for larger models
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Set model to evaluation mode
model.eval()

# Input prompt
prompt = "red riding hood"

# Tokenize the input prompt
input_ids = tokenizer.encode(prompt, return_tensors="pt")

print(input_ids)

# Generate text

output = model.generate(input_ids, max_length=100, num_return_sequences=1)

# Decode the generated output
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

# Print the generated text
print(generated_text)
